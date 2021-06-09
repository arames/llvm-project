//===- llvm/Support/HashBuilder.h - Convenient hashing interface-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an interface allowing to conveniently build hashes of
// various data types, without relying on the underlying hasher type to know
// about hashed data types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HASHBUILDER_H
#define LLVM_SUPPORT_HASHBUILDER_H

#include "llvm/ADT/ArrayRef.h"
//#include "llvm/ADT/StringRef.h"
#include "llvm/Support/type_traits.h"

#include <string>
#include <utility>

namespace llvm {

namespace hash_builder {
namespace detail {
/// Trait to indicate whether a type's bits can be hashed directly.
/// A type for which this trait is true will be treated like a bag of bytes by
/// `HashBuilder`.
template <typename T>
struct is_hashable_data
    : std::integral_constant<bool, is_integral_or_enum<T>::value ||
                                       std::is_floating_point<T>::value> {};
} // namespace detail
} // namespace hash_builder

/// Interface to help hash various types through a hasher type.
///
/// The only methods expected from the templated hasher type `HasherT` are:
/// * a default constructor
/// * void update(llvm::ArrayRef<uint8_t>)
///
/// This interface provides the following:
/// * `template<typename T> update(const T &Value)`
///   The `update` function implements hashing of different types via the
///   hasher, without requiring knowledge of hashed types from the hasher.
/// * `template <typename ItT> void updateRange(ItT First, ItT Last)`
///   The `updateRange` function is designed to aid hashing a range of values.
/// * Public access to the backing hasher instance `HasherT &Hasher`.
///
/// Various specializations of `update` and `updateRange` are provided for
/// commonly used types (e.g. `ArrayRef`, `StringRef`, integral, enum, and
/// floating-point types, etc.).
///
/// This interface does not assume whether the hasher type considers the size of
/// variable-size types (e.g. `ArrayRef`, `StringRef`) as part of the hash.
/// For safety and convenience, the interface thus explicitly updates the hash
/// with the size to avoid collisions. See specializations of `update` and
/// `updateRange` for details.
///
/// User-defined `struct` types can participate in this interface by providing a
/// `updateHash` templated function. See the associated template specialization
/// for details.
template <typename HasherT> class HashBuilder {
public:
  HasherT &Hasher;

  explicit HashBuilder(HasherT &Hasher) : Hasher(Hasher) {}

protected:
  template <typename T>
  using has_update_hash_t =
      decltype(updateHash(std::declval<HashBuilder &>(), std::declval<T &>()));

public:
  /// Implement hashing for hashable data types, e.g. integral, enum, or
  /// floating-point values.
  ///
  /// The value is simply treated like a bag of bytes.
  template <class T>
  std::enable_if_t<hash_builder::detail::is_hashable_data<T>::value>
  update(T Value) {
    updateRaw(
        ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(&Value), sizeof(Value)));
  }

  /// Implement hashing for user-defined `struct`s.
  ///
  /// Any user-define `struct` can participate in hashing via `HashBuilder` by
  /// providing a `updateHash` templated function:
  //
  /// ```
  /// template <typename HasherT>
  /// void updateHash(HashBuilder<HasherT> &HBuilder,
  ///                 const UserDefinedStruct &Value);
  /// ```
  ///
  /// For example:
  ///
  /// ```
  /// struct SimpleStruct { char c; int i; };
  //
  /// template <typename HasherT>
  /// void updateHash(HashBuilder<HasherT> &HBuilder,
  ///                 const SimpleStruct &Value) {
  ///   HBuilder.update(Value.c);
  ///   HBuilder.update(Value.i);
  /// }
  ///
  /// struct StructWithPrivateMember {
  ///  public:
  ///   explicit StructWithPrivateMember(int i, float f) : i(i), f(f) {}
  ///
  ///   int i;
  ///  private:
  ///   float f;
  ///
  ///   template <typename HasherT>
  ///   friend void updateHash(HashBuilder<HasherT> &HBuilder,
  ///                          const StructWithPrivateMember& Value) {
  ///     HBuilder.update(Value.i);
  ///     HBuilder.update(Value.f);
  ///   }
  /// };
  /// ```
  template <class T>
  std::enable_if_t<is_detected<has_update_hash_t, T>::value>
  update(const T &Value) {
    updateHash(*this, Value);
  }

  /// Implement hashing of `ArrayRef<T>` for hashable data types, e.g. integral,
  /// enum, or floating-point values.
  ///
  /// `Value.size()` is taken into account, to ensure we differentiate cases
  /// like:
  /// ```
  /// builder.update({1});
  /// builder.update({2, 3});
  /// ```
  /// and
  /// ```
  /// builder.update({1, 2});
  /// builder.update({3});
  /// ```
  template <typename T>
  std::enable_if_t<hash_builder::detail::is_hashable_data<T>::value>
  update(ArrayRef<T> Value) {
    update(Value.size());
    updateRaw(
        llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Value.data()),
                                Value.size() * sizeof(T)));
  }

  template <typename T>
  std::enable_if_t<!hash_builder::detail::is_hashable_data<T>::value>
  update(ArrayRef<T> Value) {
    updateRange(Value.begin(), Value.end());
  }

  /// Implement hashing of `StringRef`.
  ///
  /// `Value.size()` is taken into account, to ensure we differentiate cases
  /// like:
  /// ```
  /// builder.update("a");
  /// builder.update("bc");
  /// ```
  /// and
  /// ```
  /// builder.update("ab");
  /// builder.update("c");
  /// ```
  //void update(StringRef Value) {
  //  update(llvm::ArrayRef<uint8_t>(
  //      reinterpret_cast<const uint8_t *>(Value.data()), Value.size()));
  //}

  template <typename T> void update(const std::basic_string<T> &Value) {
    return updateRange(Value.begin(), Value.end());
  }

  template <typename T1, typename T2> void update(std::pair<T1, T2> Value) {
    update(Value.first);
    update(Value.second);
  }

  template <typename... Ts>
  typename std::enable_if<(sizeof...(Ts) > 1)>::type
  update(std::tuple<Ts...> Arg) {
    updateTupleHelper(Arg, typename std::index_sequence_for<Ts...>());
  }

  /// A convenenience variadic helper.
  /// It simply iterates over its arguments, in order.
  /// ```
  /// update(Arg1, Arg2);
  /// ```
  /// is equivalent to
  /// ```
  /// update(Arg1)
  /// update(Arg2)
  /// ```
  template <typename... Ts>
  typename std::enable_if<(sizeof...(Ts) > 1)>::type update(const Ts &...Args) {
    std::tuple<const Ts &...> Unused{(update(Args), Args)...};
  }

  template <typename InputIteratorT>
  void updateRange(InputIteratorT First, InputIteratorT Last) {
    updateRangeImpl(First, Last);
  }

private:
  /// Update the hash with a "bag of bytes". This does not rely on `HasherT` to
  /// take `Value.size()` into account.
  void updateRaw(ArrayRef<uint8_t> Value) { Hasher.update(Value); }

  template <typename... Ts, std::size_t... Indices>
  void updateTupleHelper(const std::tuple<Ts...> &Arg,
                         std::index_sequence<Indices...>) {
    std::tuple<const Ts &...> Unused{
        (update(std::get<Indices>(Arg)), std::get<Indices>(Arg))...};
  }

  template <typename T>
  std::enable_if_t<hash_builder::detail::is_hashable_data<T>::value>
  updateRangeImpl(T *First, T *Last) {
    update(Last - First);
    updateRaw(llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(First),
                                      (Last - First) * sizeof(T)));
  }

  template <typename InputIteratorT>
  void updateRangeImpl(InputIteratorT First, InputIteratorT Last) {
    update(Last - First);
    for (auto It = First; It != Last; ++It)
      update(*It);
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_HASHBUILDER_H
