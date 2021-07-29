//===- JoinMeetTypeInterface.cpp - Join/Meet Type Interface Implementation ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/JoinMeetTypeInterface.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

#include "mlir/Interfaces/JoinMeetTypeInterface.cpp.inc"

Type mlir::joinTypes(Type ty1, Type ty2) {
  if (ty1 == ty2)
    return ty1;
  if (!ty1 || !ty2)
    return Type();

  Optional<Type> join1;
  Optional<Type> join2;

  if (auto interface1 = ty1.dyn_cast<JoinMeetTypeInterface>())
    join1 = interface1.join(ty2);

#ifndef NDEBUG
  static constexpr bool assertCompatibility = true;
#else
  static constexpr bool assertCompatibility = false;
#endif

  auto interface2 = ty2.dyn_cast<JoinMeetTypeInterface>();
  if (interface2 && (!join1 || assertCompatibility))
    join2 = interface2.join(ty1);

  assert((!assertCompatibility ||
          (!join1 || !join2 || join1.getValue() == join2.getValue())) &&
         "joinTypes commutativity was violated");

  return join1 ? join1.getValue() : join2.getValueOr(Type());
}

Type mlir::meetTypes(Type ty1, Type ty2) {
  if (ty1 == ty2)
    return ty1;
  if (!ty1 || !ty2)
    return Type();

  Optional<Type> meet1;
  if (auto interface1 = ty1.dyn_cast<JoinMeetTypeInterface>())
    meet1 = interface1.meet(ty2);

  Optional<Type> meet2;
  if (auto interface2 = ty2.dyn_cast<JoinMeetTypeInterface>())
    meet2 = interface2.meet(ty1);

  assert(!meet1 || !meet2 || meet1.getValue() == meet2.getValue());

  return meet1 ? meet1.getValue() : meet2.getValueOr(Type());
}
