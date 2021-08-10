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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/type_traits.h"

#include <iterator>
#include <utility>

namespace llvm {

/// Interface to help hash various types through a hasher type.
///
/// Via provided specializations of `update` and `updateRange` functions,
/// various types (e.g. `ArrayRef`, `StringRef`, etc.) can be hashed without
/// requiring any knowledge of hashed types from the hasher type.
///
/// The only methods expected from the templated hasher type `HasherT` are:
/// * a default constructor
/// * void update(ArrayRef<uint8_t> Data)
///
/// From a user point of view, the interface provides the following:
/// * `template<typename T> update(const T &Value)`
///   The `update` function implements hashing of various types.
/// * `template <typename ItT> void updateRange(ItT First, ItT Last)`
///   The `updateRange` function is designed to aid hashing a range of values.
///
/// User-defined `struct` types can participate in this interface by providing
/// an `updateHash` templated function. See the associated template
/// specialization for details.
///
/// This interface does not impose requirements on the hasher
/// `update(const uint8_t *Ptr, size_t Size)` method.
/// We want to avoid collisions for variable-size types; for example for
/// ```
/// builder.update({1});
/// builder.update({2, 3});
/// ```
/// and
/// ```
/// builder.update({1, 2});
/// builder.update({3});
/// ```
/// . Thus, specializations of `update` and `updateHash` for variable-size types
/// must not assume that the hasher type considers the size as part of the
/// hash; they must explicitly update the hash with the size. See for example
/// specializations for `ArrayRef` and `StringRef`.
///
/// Additionally, since types are eventually forwarded to the hasher's
/// `void update(ArrayRef<uint8_t>)` method, endianness plays a role in the hash
/// computation (for example when computing `update((int)123)`).
/// Specifying endianness via the `Endianness` template parameter allows to
/// compute stable hash across platforms with different endianness.
template <typename HasherT,
          support::endianness Endianness = support::endianness::native>
class HashBuilder {
private:
  /// Trait to indicate whether a type's bits can be hashed directly (after
  /// endianness correction).
  template <typename U>
  struct IsHashableData
      : std::integral_constant<bool, is_integral_or_enum<U>::value ||
                                         std::is_floating_point<U>::value> {};

public:
  Optional<HasherT> OptionalHasher;
  HasherT &Hasher;

  explicit HashBuilder(HasherT &Hasher) : Hasher(Hasher) {}
  template <typename... ArgTypes>
  explicit HashBuilder(ArgTypes &&...Args)
      : OptionalHasher(in_place, std::forward<ArgTypes>(Args)...),
        Hasher(*OptionalHasher) {}

  template <typename T>
  using has_final_t = decltype(std::declval<T &>().final());
  template <
      typename E = std::enable_if_t<is_detected<has_final_t, HasherT>::value>>
  auto final() {
    return Hasher.final();
  }

  /// Implement hashing for hashable data types, e.g. integral, enum, or
  /// floating-point values.
  template <typename T>
  std::enable_if_t<IsHashableData<T>::value, HashBuilder &> update(T Value) {
    Value = support::endian::byte_swap(Value, Endianness);
    return updateBytes(
        makeArrayRef(reinterpret_cast<const uint8_t *>(&Value), sizeof(Value)));
  }

  /// Support hashing `ArrayRef`.
  ///
  /// `Value.size()` is taken into account to ensure cases like
  /// ```
  /// builder.update({1});
  /// builder.update({2, 3});
  /// ```
  /// and
  /// ```
  /// builder.update({1, 2});
  /// builder.update({3});
  /// ```
  /// do not collide.
  template <typename T> HashBuilder &update(ArrayRef<T> Value) {
    return updateRange(Value);
  }

  /// Support hashing `StringRef`.
  ///
  /// `Value.size()` is taken into account to ensure cases like
  /// ```
  /// builder.update("a");
  /// builder.update("bc");
  /// ```
  /// and
  /// ```
  /// builder.update("ab");
  /// builder.update("c");
  /// ```
  /// do not collide.
  HashBuilder &update(StringRef Value) { return updateRange(Value); }

  /// Implement hashing for user-defined `struct`s.
  ///
  /// Any user-define `struct` can participate in hashing via `HashBuilder` by
  /// providing a `updateHash` templated function.
  ///
  /// ```
  /// template <typename HasherT, support::endianness Endianness>
  /// void updateHash(HashBuilder<HasherT, Endianness> &HBuilder,
  ///                 const UserDefinedStruct &Value);
  /// ```
  ///
  /// For example:
  /// ```
  /// struct SimpleStruct {
  ///   char c;
  ///   int i;
  /// };
  ///
  /// template <typename HasherT, support::endianness Endianness>
  /// void updateHash(HashBuilder<HasherT, Endianness> &HBuilder,
  ///                 const SimpleStruct &Value) {
  ///   HBuilder.update(Value.c);
  ///   HBuilder.update(Value.i);
  /// }
  /// ```
  ///
  /// To avoid endianness issues, specializations of `updateHash` should
  /// generally rely on exising `update` and `updateRange` functions. If
  /// directly using `updateBytes`, an implementation must correctly handle
  /// endianness.
  ///
  /// ```
  /// struct __attribute__ ((packed)) StructWithFastHash {
  ///   int I;
  ///   char C;
  ///
  ///   // If possible, we want to hash both `I` and `C` in a single
  ///   `updateBytes`
  ///   // call for performance concerns.
  ///   template <typename HasherT, support::endianness Endianness>
  ///   friend void updateHash(HashBuilder<HasherT, Endianness> &HBuilder,
  ///                          const StructWithFastHash &Value) {
  ///     if (Endianness == support::endianness::native ||
  ///         Endianness == support::endian::system_endianness()) {
  ///       HBuilder.updateBytes(makeArrayRef(
  ///           reinterpret_cast<const uint8_t *>(&Value), sizeof(Value)));
  ///     } else {
  ///       // Rely on existing `update` methods to handle endianness.
  ///       HBuilder.update(Value.I);
  ///       HBuilder.update(Value.C);
  ///     }
  ///   }
  /// };
  /// ```
  ///
  /// To avoid collisions, specialization of `updateHash` for variable-size
  /// types must take the size into account.
  ///
  /// For example:
  /// ```
  /// struct CustomContainer {
  /// private:
  ///   size_t Size;
  ///   int Elements[100];
  ///
  /// public:
  ///   CustomContainer(size_t Size) : Size(Size) {
  ///     for (size_t I = 0; I != Size; ++I)
  ///       Elements[I] = I;
  ///   }
  ///   template <typename HasherT, support::endianness Endianness>
  ///   friend void updateHash(HashBuilder<HasherT, Endianness> &HBuilder,
  ///                          const CustomContainer &Value) {
  ///     if (Endianness == support::endianness::native ||
  ///         Endianness == support::endian::system_endianness()) {
  ///       HBuilder.updateBytes(makeArrayRef(
  ///           reinterpret_cast<const uint8_t *>(&Value.Size),
  ///           sizeof(Value.Size) + Value.Size * sizeof(Value.Elements[0])));
  ///     } else {
  ///       // `updateRange` will take care of encoding the size.
  ///       HBuilder.updateRange(&Value.Elements[0], &Value.Elements[0] +
  ///       Value.Size);
  ///     }
  ///   }
  /// };
  /// ```
  ///
  template <typename T>
  using has_update_hash_t =
      decltype(updateHash(std::declval<HashBuilder &>(), std::declval<T &>()));
  template <typename T>
  std::enable_if_t<is_detected<has_update_hash_t, T>::value, HashBuilder &>
  update(const T &Value) {
    updateHash(*this, Value);
    return *this;
  }

  template <typename T1, typename T2>
  HashBuilder &update(const std::pair<T1, T2> &Value) {
    update(Value.first);
    update(Value.second);
    return *this;
  }

  template <typename... Ts>
  typename std::enable_if<(sizeof...(Ts) > 1), HashBuilder &>::type
  update(const std::tuple<Ts...> &Arg) {
    return updateTupleHelper(Arg, typename std::index_sequence_for<Ts...>());
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
  typename std::enable_if<(sizeof...(Ts) > 1), HashBuilder &>::type
  update(const Ts &...Args) {
    std::tuple<const Ts &...>{(update(Args), Args)...};
    return *this;
  }

  template <typename ForwardIteratorT>
  HashBuilder &updateRange(ForwardIteratorT First, ForwardIteratorT Last) {
    return updateRangeImpl(
        First, Last,
        typename std::iterator_traits<ForwardIteratorT>::iterator_category());
  }

  template <typename RangeT> HashBuilder &updateRange(const RangeT &Range) {
    return updateRange(adl_begin(Range), adl_end(Range));
  }

  /// Update the hash with `Data`. This does not rely on `HasherT` to
  /// take the size of `Data` into account.
  ///
  /// Users of this function should pay attention to respect endianness
  /// contraints.
  HashBuilder &updateBytes(ArrayRef<uint8_t> Data) {
    Hasher.update(Data);
    return *this;
  }

  /// Update the hash with `Data`. This does not rely on `HasherT` to
  /// take the size of `Data` into account.
  ///
  /// Users of this function should pay attention to respect endianness
  /// contraints.
  HashBuilder &updateBytes(StringRef Data) {
    return updateBytes(makeArrayRef(
        reinterpret_cast<const uint8_t *>(Data.data()), Data.size()));
  }

private:
  template <typename... Ts, std::size_t... Indices>
  HashBuilder &updateTupleHelper(const std::tuple<Ts...> &Arg,
                                 std::index_sequence<Indices...>) {
    std::tuple<const Ts &...>{
        (update(std::get<Indices>(Arg)), std::get<Indices>(Arg))...};
    return *this;
  }

  // FIXME: Once available, specialize this function for `contiguous_iterator`s,
  // and use it for `ArrayRef` and `StringRef`.
  template <typename ForwardIteratorT>
  HashBuilder &updateRangeImpl(ForwardIteratorT First, ForwardIteratorT Last,
                               std::forward_iterator_tag) {
    update(std::distance(First, Last));
    for (auto It = First; It != Last; ++It)
      update(*It);
    return *this;
  }

  template <typename T>
  std::enable_if_t<IsHashableData<T>::value &&
                       Endianness == support::endian::system_endianness(),
                   HashBuilder &>
  updateRangeImpl(T *First, T *Last, std::forward_iterator_tag) {
    update(std::distance(First, Last));
    updateBytes(makeArrayRef(reinterpret_cast<const uint8_t *>(First),
                             (Last - First) * sizeof(T)));
    return *this;
  }
};

} // end namespace llvm

#endif // LLVM_SUPPORT_HASHBUILDER_H
