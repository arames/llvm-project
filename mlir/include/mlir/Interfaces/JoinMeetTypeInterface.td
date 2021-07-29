//===- JoinMeetTypeInterface.td - Join/Meet Type Interface ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for join and meet type relathionships.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_JOINMEETTYPEINTERFACE
#define MLIR_JOINMEETTYPEINTERFACE

include "mlir/IR/OpBase.td"

def JoinMeetTypeInterface : TypeInterface<"JoinMeetTypeInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    Interface for types participating in the `join` and `meet` relationships.

    Types willing to participate in the `::mlir::joinTypes(Type, Type)` and
    `::mlir::meet(Type, Type)` functions should implement this interface by
    providing implementations of `join` and `meet` functions handling types they
    can join or meet with.

    The return value of the interface method `someType.join(otherType)` is
    interpreted as follows:
    - A result with a value (`Optional<Type>::hasValue()` is true) represents
      the result of the `join`.
      If the contained type is null, it indicates `otherType` was handled, and
      `someType` and `otherType` do not join; that is, there is no type that is
      both less specialized than `someType` and less specialized than
      `otherType`.
    - A result without a value (`Optional<Type>::hasValue()` is false) indicates
      `sometype` does not know (or ignores) the `other` type.

    Returning an `Optional<Type>` with no value allows for extensibility of
    types. `Type1::join` may handle `Type2` while `Type2::join` ignores `Type1`
    or simply does not implement this interface.

    The interface `join` method must participate to guarantee the monotonicity
    of `::mlir::join`. It must be idempotent (`joinTypes(x, x) = x`).
    Associativity, though it should flow naturally, must be maintained.
    Commutativity is guaranteed by `::mlir::join` itself, as it queries both
    `sometype.join(othertype)` and `othertype.join(sometype)`. One type may
    ignore the other; in that case the result with a value is used. If both
    results have a value, they must match.

    The same comments apply to `meet`.

    Note: for context about join/meet operations on lattices, see:
    https://en.wikipedia.org/wiki/Lattice_(order).
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Return the join with the `other` type.",
      /*retTy=*/"Optional<Type>",
      /*methodName=*/"join",
      /*args=*/(ins "Type":$other)
    >,
    InterfaceMethod<
      /*description=*/"Return the meet with the `other` type.",
      /*retTy=*/"Optional<Type>",
      /*methodName=*/"meet",
      /*args=*/(ins "Type":$other)
    >,
  ];
}

#endif // MLIR_JOINMEETTYPEINTERFACE
