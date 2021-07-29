//===- JoinMeetTypeInterface.h - Join/Meet Type Interface Decls -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for type join and meet relationships.
// Types can inherit and implement the `JoinMeetTypeInterface::Trait` trait to
// participate in `join` and `meet` functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_JOINMEETTYPEINTERFACE_H
#define MLIR_INTERFACES_JOINMEETTYPEINTERFACE_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"

#include "mlir/Interfaces/JoinMeetTypeInterface.h.inc"

namespace mlir {

//===----------------------------------------------------------------------===//
// Join and Meet Types
//===----------------------------------------------------------------------===//

/// The join function for types, and the partial order "less specialized than or
/// equal".
/// It returns the most specialized type that is less specialized than both
/// `ty1` and `ty2`, if it exists, and `Type()` otherwise.
/// The join `j` of `ty1` and `ty2` is such that:
/// * j <= ty1, and j <= ty2
/// * For any type t such that t <= ty1 and t <= ty2, t <= j.
///
/// In other words, if types are viewed as sets of values, this function is
/// equivalent to the union of such sets. The top of the lattice represents "any
/// possible type", and the bottom represents "no possible type".
///
/// For example:
///  ty1               | ty2               | ty1 v ty2
///  ------------------+-------------------+-------------------
///  i8                | i8                | i8
///  i8                | i32               | <none> (null type)
///  tensor<1xf32>     | tensor<?xf32>     | tensor<?xf32>
///  tensor<1x2x?xf32> | tensor<1x?x3xf32> | tensor<1x?x?xf32>
///  tensor<4x5xf32>   | tensor<6xf32>     | tensor<*xf32>
///  tensor<1xi32>     | i32               | <none> (null type)
///  tensor<1xi32>     | tensor<i32>       | tensor<*xi32>
///  tensor<1xi32>     | tensor<1xi8>      | <none> (null type)
///
/// The function is monotonic:
/// * idempotence:   joinTypes(x,x) == x
/// * commutativity: joinTypes(x,y) == joinTypes(y,x)
/// * associativity: joinTypes(x,joinTypes(y,z)) == joinTypes(joinTypes(x,y),z)
///
/// Types can participate in this function by implementing
/// `JoinMeetTypeInterface`.
Type joinTypes(Type ty1, Type ty2);

/// The meet function for types, and the partial order "less specialized than or
/// equal".
/// It returns the least specialized type, that is more specialized than both
/// `ty1` and `ty2`, if it exists, and `Type()` otherwise.
/// The meet `m` of `ty1` and `ty2` is such that:
/// * ty1 <= m, and ty2 <= m
/// * For any type t such that ty1 <= t and ty2 <= t, m <= t.
///
/// In other words, if types are viewed as sets of values, this function is
/// equivalent to the intersection of such sets. The top of the lattice
/// represents "any possible type", and the bottom represents "no possible
/// type".
///
/// For example:
///  ty1               | ty2               | ty1 ^ ty2
///  ------------------+-------------------+-------------------
///  i8                | i32               | <none> (null type)
///  tensor<1xf32>     | tensor<?xf32>     | tensor<1xf32>
///  tensor<1x2x?xf32> | tensor<1x?x3xf32> | tensor<1x2x3xf32>
///  tensor<4x5xf32>   | tensor<6xf32>     | <none> (null type)
///  tensor<1xi32>     | i32               | <none> (null type)
///  tensor<1xi32>     | tensor<i32>       | <none> (null type)
///  tensor<1xi32>     | tensor<1xi8>      | <none> (null type)
///
/// The function is monotonic:
/// * idempotence:   meetTypes(x,x) == x
/// * commutativity: meetTypes(x,y) == meetTypes(y,x)
/// * associativity: meetTypes(x,meetTypes(y,z)) == meetTypes(meetTypes(x,y),z)
///
/// Types can participate in this function by implementing
/// `JoinMeetTypeInterface`.
Type meetTypes(Type ty1, Type ty2);

/// Indicates whether `ty1` and `ty2` are the same, or `ty1` is compatible with
/// `ty2` and less specialized than `ty2`.
inline bool isLessSpecializedOrSame(Type ty1, Type ty2) {
  return joinTypes(ty1, ty2) == ty1;
}

/// Indicates whether `ty1` is compatible with `ty2`, and less specialized than
/// `ty2`.
inline bool isLessSpecialized(Type ty1, Type ty2) {
  return ty1 != ty2 && isLessSpecializedOrSame(ty1, ty2);
}

/// Indicates whether `ty1` and `ty2` are the same, or `ty1` is compatible with
/// `ty2` and more specialized than `ty2`.
inline bool isMoreSpecializedOrSame(Type ty1, Type ty2) {
  return meetTypes(ty1, ty2) == ty1;
}

/// Indicates whether `ty1` is compatible with `ty2`, and more specialized than
/// `ty2`.
inline bool isMoreSpecialized(Type ty1, Type ty2) {
  return ty1 != ty2 && isMoreSpecializedOrSame(ty1, ty2);
}

} // namespace mlir

#endif // MLIR_INTERFACES_JOINMEETTYPEINTERFACE_H
