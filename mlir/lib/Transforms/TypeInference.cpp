//===- TypeInference.cpp - Infer Types of MLIR operations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/JoinMeetTypeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TypeLatticeValue {
  TypeLatticeValue() = default;
  TypeLatticeValue(Type type) : type(type) {}

  static TypeLatticeValue getPessimisticValueState(MLIRContext *context) {
    return TypeLatticeValue();
  }

  static TypeLatticeValue getPessimisticValueState(Value value) {
    return TypeLatticeValue();
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

struct TypeInferenceAnalysis : public ForwardDataFlowAnalysis<TypeLatticeValue> {
public:
  using ForwardDataFlowAnalysis<TypeLatticeValue>::ForwardDataFlowAnalysis;
  TypeInferenceAnalysis(MLIRContext *context)
      : ForwardDataFlowAnalysis(context), status(success()) {}
  ~TypeInferenceAnalysis() override = default;

  ChangeResult visitOperation(
      Operation *op,
      ArrayRef<LatticeElement<TypeLatticeValue> *> operands) override {
    // FIXME: We may want a way to interrupt the analysis when something went
    // wrong.
    if (failed(status))
      return ChangeResult::NoChange;

    auto interface = dyn_cast<InferTypeOpInterface>(op);
    if (!interface)
      return ChangeResult::NoChange;
    SmallVector<Type, 2> inferredReturnTypes;
    status = interface.inferReturnTypes(
        op->getContext(), op->getLoc(), op->getOperands(),
        op->getAttrDictionary(), op->getRegions(), inferredReturnTypes);
    if (failed(status)) {
      op->emitError("failed to infer types");
      return ChangeResult::NoChange;
    }

    ChangeResult result = ChangeResult::NoChange;

    for (unsigned int i = 0; i < op->getNumResults(); i++) {
      LatticeElement<TypeLatticeValue> &lattice =
          getLatticeElement(op->getResult(i));
      result |= lattice.join(inferredReturnTypes[i]);
    }

    return result;
  }

  LogicalResult getStatus() const { return status; }

private:
  LogicalResult status;
};
} // end anonymous namespace

static void updateTypes(TypeInferenceAnalysis &analysis, Operation *op) {
  // TODO:
  // This is extremely simplified.
  // Not all ops will allow refining input types.
  op->walk([&](mlir::Operation *op) {
    for (Value res : op->getResults()) {
      LatticeElement<TypeLatticeValue> *latticeElement =
          analysis.lookupLatticeElement(res);
      if (latticeElement)
        res.setType(latticeElement->getValue().type);
    }
  });
}

namespace {
struct TypeInference : public TypeInferencePassBase<TypeInference> {
  void runOnOperation() override {
    Operation *op = getOperation();
    TypeInferenceAnalysis analysis(op->getContext());
    analysis.run(op);
    updateTypes(analysis, op);
  }
};

} // end anonymous namespace

/// Create a TypeInference pass.
std::unique_ptr<Pass> mlir::createTypeInferencePass() {
  return std::make_unique<TypeInference>();
}
