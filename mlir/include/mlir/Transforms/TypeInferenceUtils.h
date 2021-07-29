//===- TypeInferenceUtils.h - Type inference utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_TYPEINFERENCEUTILS_H
#define MLIR_TRANSFORMS_TYPEINFERENCEUTILS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
using TypeInferenceTypeAdapter = std::function<Value(Value, Type)>;

/// The default type adapter used by type inference, generating `std.relax_type`
/// or `std.specialize_type` ops.
Value defaultTypeInferenceAdapter(Value value, Type type);
} // end namespace mlir

#endif // MLIR_TRANSFORMS_TYPEINFERENCEUTILS_H
