//===- TestTypeInferenceAdapter.cpp - Test TI WITH Configurable Adapt  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
Value customTypeAdapter(Value value, Type type) {
  if (type.isa<TensorType>()) {
    OpBuilder builder(value.getContext());
    builder.setInsertionPointAfterValue(value);
    return builder.createOrFold<tensor::CastOp>(value.getLoc(), type, value);
  }
  return defaultTypeInferenceAdapter(value, type);
}

struct TestTypeInferenceAdapter
    : public PassWrapper<TestTypeInferenceAdapter, Pass> {
  StringRef getArgument() const final { return "test-type-inference-adapter"; }
  StringRef getDescription() const final {
    return "Test type inference with a configurable type adapter";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override {
    PassManager passManager(getOperation()->getContext());
    passManager.addPass(createTypeInferencePass(customTypeAdapter));
    if (failed(passManager.run(getOperation())))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestTypeInferenceAdapter() {
  PassRegistration<TestTypeInferenceAdapter>();
}
} // namespace test
} // namespace mlir
