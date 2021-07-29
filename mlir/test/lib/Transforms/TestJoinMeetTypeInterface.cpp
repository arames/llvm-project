//===- TestJoinMeetTypeInterface.cpp - Test The Join/Meet Type Interface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Interfaces/JoinMeetTypeInterface.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct TestJoinMeetTypeInterface
    : public PassWrapper<TestJoinMeetTypeInterface, FunctionPass> {
  StringRef getArgument() const final {
    return "test-join-meet-type-interface";
  }
  StringRef getDescription() const final {
    return "Test join/meet type interfaceTest operation constant folding";
  }
  void runOnFunction() override {
    FuncOp func = getFunction();
    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "join") {
        Type join =
            joinTypes(op->getOperand(0).getType(), op->getOperand(1).getType());
        if (join)
          op->getResult(0).setType(join);
        else
          op->emitError("types do not join");
      }
      if (op->getName().getStringRef() == "meet") {
        Type meet =
            meetTypes(op->getOperand(0).getType(), op->getOperand(1).getType());
        if (meet)
          op->getResult(0).setType(meet);
        else
          op->emitError("types do not meet");
      }
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestJoinMeetTypeInterface() {
  PassRegistration<TestJoinMeetTypeInterface>();
}
} // namespace test
} // namespace mlir
