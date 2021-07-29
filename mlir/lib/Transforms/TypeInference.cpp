//===- TypeInference.cpp - Infer Types of MLIR operations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The type inference analysis uses the dataflow analysis infrastructure to
// infer MLIR value types over the IR.
// The type inference pass uses the results of the analysis to update MLIR value
// types, querying operation interfaces to appropriately update types in place
// or insert type relaxation or specialization ops.
//
// The `join` relationship over the type lattice is described more in details in
// `JoinMeetTypeInterface.h`. The type lattice looks like
//
//    join(t1, t2)
//    tensor<?x?x3xf32>
//    /              \
//   /                \
//  /                  \
// t1                  t2
// tensor<1x?x3xf32>   tensor<?x2x3xf32>
//  \                  /
//   \                /
//    \              /
//    meet(t1, t2)
//    tensor<1x2x3xf32>
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/JoinMeetTypeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "infer-types"

using namespace mlir;
using llvm::dbgs;

namespace {
struct TypeLatticeValue {
  TypeLatticeValue() = default;
  TypeLatticeValue(Type type) : type(type) {}

  static TypeLatticeValue getPessimisticValueState(MLIRContext *context) {
    return TypeLatticeValue();
  }

  static TypeLatticeValue getPessimisticValueState(Value value) {
    return value.getType();
  }

  static TypeLatticeValue join(const TypeLatticeValue &lhs,
                               const TypeLatticeValue &rhs) {
    return joinTypes(lhs.type, rhs.type);
  }

  bool operator==(const TypeLatticeValue &rhs) const {
    return type == rhs.type;
  }

  Type type;
};

struct TypeInferenceAnalysis
    : public ForwardDataFlowAnalysis<TypeLatticeValue> {
public:
  using ForwardDataFlowAnalysis<TypeLatticeValue>::ForwardDataFlowAnalysis;
  TypeInferenceAnalysis(MLIRContext *context)
      : ForwardDataFlowAnalysis(context), status(success()) {}
  ~TypeInferenceAnalysis() override = default;

  ChangeResult visitOperation(
      Operation *op,
      ArrayRef<LatticeElement<TypeLatticeValue> *> operands) override;

  LogicalResult getStatus() const { return status; }

private:
  // A wrapper allowing tracking updates for debugging purposes.
  ChangeResult updateLatticeElement(Value value, Type ty) {
    LatticeElement<TypeLatticeValue> &latticeElement = getLatticeElement(value);
    LLVM_DEBUG(dbgs() << "update lattice element for :\t"; value.print(dbgs());
               dbgs() << "\n\toriginal type:\t";
               if (latticeElement.isUninitialized()) dbgs() << "uninitialized";
               else latticeElement.getValue().type.print(dbgs());
               dbgs() << "\n\tjoining with:\t"; ty.print(dbgs()););
    ChangeResult change = latticeElement.join(ty);
    LLVM_DEBUG(dbgs() << "\n\tyielding:\t";
               latticeElement.getValue().type.print(dbgs()); dbgs() << "\n");
    return change;
  }

  LogicalResult status;
};

ChangeResult TypeInferenceAnalysis::visitOperation(
    Operation *op, ArrayRef<LatticeElement<TypeLatticeValue> *> operands) {
  // Do not continue with the analysis if something went wrong.
  // TODO: Having a way to interrupt the analysis on error would be
  // convenient.
  if (LLVM_UNLIKELY(failed(status)))
    return ChangeResult::NoChange;

  // FIXME: Have `SameOperandsAndResultType` and other similar traits provide a
  // default implementation of `InferTypeOpInterface`, so that they are
  // automatically handled here.
  auto interface = dyn_cast<InferTypeOpInterface>(op);
  if (!interface) {
    // By default, simply use the current result types.
    ChangeResult change = ChangeResult::NoChange;
    for (Value result : op->getResults())
      change |= updateLatticeElement(result, result.getType());
    return change;
  }

  // Ephemerally override operand types with inferred types.
  SmallVector<Type> originalOperandTys;
  const unsigned numOperands = op->getNumOperands();
  originalOperandTys.resize(numOperands);
  for (unsigned i = 0; i != numOperands; ++i) {
    auto operand = op->getOperand(i);
    originalOperandTys[i] = operand.getType();
    operand.setType(operands[i]->getValue().type);
  }

  // Infer types for the operation.
  SmallVector<Type> inferredReturnTypes;
  status = interface.inferReturnTypes(
      op->getContext(), op->getLoc(), op->getOperands(),
      op->getAttrDictionary(), op->getRegions(), inferredReturnTypes);

  // Immediately restore original operand types.
  for (unsigned i = 0; i != numOperands; ++i)
    op->getOperand(i).setType(originalOperandTys[i]);

  if (failed(status)) {
    op->emitError("failed to infer types");
    return ChangeResult::NoChange;
  }

  ChangeResult change = ChangeResult::NoChange;
  for (unsigned int i = 0; i < op->getNumResults(); i++)
    change |= updateLatticeElement(op->getResult(i), inferredReturnTypes[i]);
  return change;
}
} // end anonymous namespace

namespace {
class TypeInference {
public:
  explicit TypeInference(Operation *op, TypeInferenceTypeAdapter typeAdapter)
      : op(op), analysis(op->getContext()), typeAdapter(typeAdapter) {}

  /// Analyzes and update types for the processed operation.
  LogicalResult run() {
    analysis.run(op);
    if (failed(analysis.getStatus()))
      return analysis.getStatus();
    return updateOp(op);
  }

private:
  LogicalResult updateOp(Operation *op);
  LogicalResult updateBlock(Block &block);
  LogicalResult updateOpResults(Operation *op);
  LogicalResult updateType(Value result);

  Operation *op;
  TypeInferenceAnalysis analysis;
  TypeInferenceTypeAdapter typeAdapter;
};

LogicalResult TypeInference::updateOp(Operation *op) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      if (failed(updateBlock(block)))
        return failure();
    }
  }
  return updateOpResults(op);
}

LogicalResult TypeInference::updateBlock(Block &block) {
  for (auto blockArg : block.getArguments())
    if (failed(updateType(blockArg)))
      return failure();
  for (auto &nestedOp : block)
    if (failed(updateOp(&nestedOp)))
      return failure();
  return success();
}

LogicalResult TypeInference::updateOpResults(Operation *op) {
  for (auto result : op->getResults())
    if (failed(updateType(result)))
      return failure();
  return success();
}

LogicalResult TypeInference::updateType(Value value) {
  LatticeElement<TypeLatticeValue> *latticeElement =
      analysis.lookupLatticeElement(value);
  if (!latticeElement)
    return success();

  Type originalTy = value.getType();
  Type inferredTy = latticeElement->getValue().type;
  if (inferredTy == originalTy)
    return success();

  Operation *definingOp = value.getDefiningOp();

  if (!isMoreSpecialized(inferredTy, originalTy))
    return emitError(value.getLoc(), "inferred type ")
           << inferredTy << " is not more specialized than the original type"
           << originalTy;

  // Values and uses may or may not allow specializing input or output types.
  // Keep track of the "value" with its original type, and after type
  // specialization.
  Value valueWithOriginalType = value;
  Value valueWithInferredType = {};

  // Attempt to specialize the output type.
  auto interface =
      dyn_cast_or_null<AllowsOutputTypesSpecializationInterface>(definingOp);
  bool explicitlyAllowedOutputTypeSpecialization =
      interface && interface.allowsOutputTypeSpecialization(
                       value.cast<OpResult>().getResultNumber());
  bool implicitlyAllowedOutputTypeSpecialization =
      !interface && dyn_cast_or_null<InferTypeOpInterface>(definingOp);
  if (explicitlyAllowedOutputTypeSpecialization ||
      implicitlyAllowedOutputTypeSpecialization) {
    value.setType(inferredTy);
    valueWithOriginalType = {};
    valueWithInferredType = value;
  }

  // Update uses with the appropriate "value", specialized or not.
  unsigned nSpecializedUses = 0;
  for (auto &use : llvm::make_early_inc_range(value.getUses())) {
    auto interface =
        dyn_cast<AllowsInputTypesSpecializationInterface>(use.getOwner());
    bool allowsInputTypeSpecialization =
        interface &&
        interface.allowsInputTypeSpecialization(use.getOperandNumber());
    if (allowsInputTypeSpecialization) {
      if (!valueWithInferredType) {
        valueWithInferredType = typeAdapter(value, inferredTy);
        if (!valueWithInferredType)
          return emitError(value.getLoc(),
                           "type adapter failed to specialize the type of ")
                 << value;
      }
      use.set(valueWithInferredType);
      ++nSpecializedUses;
    } else {
      if (!valueWithOriginalType) {
        valueWithOriginalType = typeAdapter(value, originalTy);
        if (!valueWithOriginalType)
          return emitError(value.getLoc(),
                           "type adapter failed to relax the type of ")
                 << value;
      }
      use.set(valueWithOriginalType);
    }
  }

  LLVM_DEBUG(dbgs() << "updated type from "; originalTy.print(dbgs());
             dbgs() << " to "; inferredTy.print(dbgs());
             dbgs() << llvm::format(
                 " (with %u/%u specialized uses) for ", nSpecializedUses,
                 std::distance(value.getUses().begin(), value.getUses().end()));
             value.print(dbgs()); dbgs() << "\n");

  return success();
}

class TypeInferencePass : public TypeInferencePassBase<TypeInferencePass> {
public:
  TypeInferencePass(TypeInferenceTypeAdapter typeAdapter)
      : typeAdapter(typeAdapter) {}

  void runOnOperation() override {
    TypeInference typeInference(getOperation(), typeAdapter);
    if (failed(typeInference.run()))
      signalPassFailure();
  }

private:
  TypeInferenceTypeAdapter typeAdapter;
};
} // end anonymous namespace

Value mlir::defaultTypeInferenceAdapter(Value value, Type type) {
  Type currentType = value.getType();
  if (currentType == type)
    return value;
  OpBuilder builder(value.getContext());
  builder.setInsertionPointAfterValue(value);
  if (isLessSpecialized(currentType, type))
    return builder.createOrFold<SpecializeTypeOp>(value.getLoc(), value, type);
  if (isMoreSpecialized(currentType, type))
    return builder.createOrFold<RelaxTypeOp>(value.getLoc(), value, type);
  return {};
}

std::unique_ptr<Pass>
mlir::createTypeInferencePass(TypeInferenceTypeAdapter typeAdapter) {
  return std::make_unique<TypeInferencePass>(typeAdapter);
}
