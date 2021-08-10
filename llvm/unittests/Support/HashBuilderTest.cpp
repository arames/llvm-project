//===- llvm/unittest/Support/HashBuilderTest.cpp - HashBuilder unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/HashBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/SHA256.h"
#include "gtest/gtest.h"

#include <list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// gtest utilities and macros rely on using a single type. So wrap both the
// hasher type and endianness.
template <typename _HasherT, llvm::support::endianness _Endianness>
struct HasherTAndEndianness {
  using HasherT = _HasherT;
  static constexpr llvm::support::endianness Endianness = _Endianness;
};
using HasherTAndEndiannessToTest =
    ::testing::Types<HasherTAndEndianness<llvm::MD5, llvm::support::big>,
                     HasherTAndEndianness<llvm::MD5, llvm::support::little>,
                     HasherTAndEndianness<llvm::MD5, llvm::support::native>,
                     HasherTAndEndianness<llvm::SHA1, llvm::support::big>,
                     HasherTAndEndianness<llvm::SHA1, llvm::support::little>,
                     HasherTAndEndianness<llvm::SHA1, llvm::support::native>,
                     HasherTAndEndianness<llvm::SHA256, llvm::support::big>,
                     HasherTAndEndianness<llvm::SHA256, llvm::support::little>,
                     HasherTAndEndianness<llvm::SHA256, llvm::support::native>>;
template <typename HasherT> class HashBuilderTest : public testing::Test {};
TYPED_TEST_SUITE(HashBuilderTest, HasherTAndEndiannessToTest);

template <typename HasherTAndEndianness>
using HashBuilder = llvm::HashBuilder<typename HasherTAndEndianness::HasherT,
                                      HasherTAndEndianness::Endianness>;

template <typename HasherTAndEndianness, typename... Ts>
static auto computeHash(const Ts &...Args) {
  return static_cast<std::string>(
      HashBuilder<HasherTAndEndianness>().update(Args...).final());
}

template <typename HasherTAndEndianness, typename... Ts>
static auto computeHashForRange(const Ts &...Args) {
  return static_cast<std::string>(
      HashBuilder<HasherTAndEndianness>().updateRange(Args...).final());
}

// All the test infrastructure relies on the variadic helpers. Test them first.
TYPED_TEST(HashBuilderTest, VariadicHelpers) {
  {
    HashBuilder<TypeParam> HBuilder;

    HBuilder.update(100);
    HBuilder.update(2.7);
    HBuilder.update("string");

    EXPECT_EQ(HBuilder.final(), computeHash<TypeParam>(100, 2.7, "string"));
  }

  {
    HashBuilder<TypeParam> HBuilder;

    std::vector<int> Vec{100, 101, 102};
    HBuilder.updateRange(Vec);

    EXPECT_EQ(HBuilder.final(), computeHashForRange<TypeParam>(Vec));
  }

  {
    HashBuilder<TypeParam> HBuilder;

    std::vector<int> Vec{200, 201, 202};
    HBuilder.updateRange(Vec.begin(), Vec.end());

    EXPECT_EQ(HBuilder.final(),
              computeHashForRange<TypeParam>(Vec.begin(), Vec.end()));
  }
}

TYPED_TEST(HashBuilderTest, ReferenceHashCheck) {
  using HE = TypeParam;
  using H = typename HE::HasherT;
  auto E = HE::Endianness == llvm::support::native
               ? llvm::support::endian::system_endianness()
               : HE::Endianness;

  char C = 'c';
  int I = 0x12345678;
  uint64_t UI64 = static_cast<uint64_t>(1) << 50;
  enum TestEnumeration { TE_One = 1, TE_Two = 2 };
  volatile int VI = 71;
  const volatile int CVI = 72;
  double D = 123.0;

  auto Hash = computeHash<HE>(C, I, UI64, TE_Two, VI, CVI, D);

  if (std::is_same<H, llvm::MD5>::value) {
    static char const *const Hashes[2] = {"43a75756bd3479b839815a6c43a3e3f7",
                                          "e8e522c9723d8e6a100364c8fc58fd1d"};
    EXPECT_EQ(Hash, Hashes[E]);
  } else if (std::is_same<H, llvm::SHA1>::value) {
    static char const *const Hashes[2] = {
        "y\xAB>\xAE\xA5\x83\x8Fl\xC2\x1FL\x8D\xC4\xEF"
        "F\xCA\x99u\xF0"
        "E",
        "\xF0\f5Yk\xCE\x94\x9A\xC5\xD2\xEE[\xFF\xAD\xDB\xA6\x8D\x18>v"};
    EXPECT_EQ(Hash, Hashes[E]);
  } else if (std::is_same<H, llvm::SHA256>::value) {
    static char const *const Hashes[2] = {
        "\xB6\x88\xDD\xC7?"
        "\xE5\x1F\x2\xD1\x2\xA9\xDB\x9EW\xFB\xEA\x98pHy\x11\x95\x9F\x93\x98\xF9"
        "n/\xC7\xBD\x1B\xC1",
        "\xF8\xC8O*\xC8z\xC1\x12\xBA\xB5\xB8\xD5\xCC\xB7"
        "1\xDE\xAF}%\xC2"
        "f\x97ITLL\x81"
        "0\xC8\xDEJj"};
    EXPECT_EQ(Hash, Hashes[E]);
  } else {
    llvm_unreachable("Missing reference test.");
  }
}

struct SimpleStruct {
  char C;
  int I;
};

template <typename HasherT, llvm::support::endianness Endianness>
void updateHash(llvm::HashBuilder<HasherT, Endianness> &HBuilder,
                const SimpleStruct &Value) {
  HBuilder.update(Value.C);
  HBuilder.update(Value.I);
}

struct StructWithoutCopyOrMove {
  int I;
  StructWithoutCopyOrMove() = default;
  StructWithoutCopyOrMove(const StructWithoutCopyOrMove &) = delete;
  StructWithoutCopyOrMove &operator=(const StructWithoutCopyOrMove &) = delete;

  template <typename HasherT, llvm::support::endianness Endianness>
  friend void updateHash(llvm::HashBuilder<HasherT, Endianness> &HBuilder,
                         const StructWithoutCopyOrMove &Value) {
    HBuilder.update(Value.I);
  }
};

struct __attribute__((packed)) StructWithFastHash {
  int I;
  char C;

  // If possible, we want to hash both `I` and `C` in a single `updateBytes`
  // call for performance concerns.
  template <typename HasherT, llvm::support::endianness Endianness>
  friend void updateHash(llvm::HashBuilder<HasherT, Endianness> &HBuilder,
                         const StructWithFastHash &Value) {
    if (Endianness == llvm::support::endianness::native ||
        Endianness == llvm::support::endian::system_endianness()) {
      HBuilder.updateBytes(llvm::makeArrayRef(
          reinterpret_cast<const uint8_t *>(&Value), sizeof(Value)));
    } else {
      // Rely on existing `update` methods to handle endianness.
      HBuilder.update(Value.I);
      HBuilder.update(Value.C);
    }
  }
};

struct CustomContainer {
private:
  size_t Size;
  int Elements[100];

public:
  CustomContainer(size_t Size) : Size(Size) {
    for (size_t I = 0; I != Size; ++I)
      Elements[I] = I;
  }
  template <typename HasherT, llvm::support::endianness Endianness>
  friend void updateHash(llvm::HashBuilder<HasherT, Endianness> &HBuilder,
                         const CustomContainer &Value) {
    if (Endianness == llvm::support::endianness::native ||
        Endianness == llvm::support::endian::system_endianness()) {
      HBuilder.updateBytes(llvm::makeArrayRef(
          reinterpret_cast<const uint8_t *>(&Value.Size),
          sizeof(Value.Size) + Value.Size * sizeof(Value.Elements[0])));
    } else {
      HBuilder.updateRange(&Value.Elements[0], &Value.Elements[0] + Value.Size);
    }
  }
};

TYPED_TEST(HashBuilderTest, HashUserDefinedStruct) {
  using HE = TypeParam;
  EXPECT_EQ(computeHash<HE>(SimpleStruct{'c', 123}), computeHash<HE>('c', 123));
  EXPECT_EQ(computeHash<HE>(StructWithoutCopyOrMove{1}), computeHash<HE>(1));
  EXPECT_EQ(computeHash<HE>(StructWithFastHash{123, 'c'}),
            computeHash<HE>(123, 'c'));
  EXPECT_EQ(computeHash<HE>(CustomContainer(3)),
            computeHash<HE>(static_cast<size_t>(3), 0, 1, 2));
}

TYPED_TEST(HashBuilderTest, HashArrayRefHashableDataTypes) {
  using HE = TypeParam;
  llvm::ArrayRef<int> Array{1, 20, 0x12345678};
  EXPECT_NE(computeHash<HE>(Array), computeHash<HE>(1, 20, 0x12345678));
  EXPECT_EQ(computeHash<HE>(Array),
            computeHashForRange<HE>(Array.begin(), Array.end()));
  EXPECT_EQ(computeHash<HE>(Array),
            computeHashForRange<HE>(Array.data(), Array.data() + Array.size()));
}

TYPED_TEST(HashBuilderTest, HashArrayRef) {
  using HE = TypeParam;
  llvm::ArrayRef<uint8_t> Array123{1, 2, 3};
  llvm::ArrayRef<uint8_t> Array12{1, 2};
  llvm::ArrayRef<uint8_t> Array1{1};
  llvm::ArrayRef<uint8_t> Array23{2, 3};
  llvm::ArrayRef<uint8_t> Array3{3};
  llvm::ArrayRef<uint8_t> ArrayEmpty{};

  auto Hash123andEmpty = computeHash<HE>(Array123, ArrayEmpty);
  auto Hash12And3 = computeHash<HE>(Array12, Array3);
  auto Hash1And23 = computeHash<HE>(Array1, Array23);
  auto HashEmptyAnd123 = computeHash<HE>(ArrayEmpty, Array123);

  EXPECT_NE(Hash123andEmpty, Hash12And3);
  EXPECT_NE(Hash123andEmpty, Hash1And23);
  EXPECT_NE(Hash123andEmpty, HashEmptyAnd123);
  EXPECT_NE(Hash12And3, Hash1And23);
  EXPECT_NE(Hash12And3, HashEmptyAnd123);
  EXPECT_NE(Hash1And23, HashEmptyAnd123);
}

TYPED_TEST(HashBuilderTest, HashArrayRefNonHashableDataTypes) {
  using HE = TypeParam;
  llvm::ArrayRef<SimpleStruct> Array{{'a', 100}, {'b', 200}};
  EXPECT_NE(computeHash<HE>(Array),
            computeHash<HE>(SimpleStruct{'a', 100}, SimpleStruct{'b', 200}));
}

TYPED_TEST(HashBuilderTest, HashStringRef) {
  using HE = TypeParam;
  llvm::StringRef SEmpty("");
  llvm::StringRef S1("1");
  llvm::StringRef S12("12");
  llvm::StringRef S123("123");
  llvm::StringRef S23("23");
  llvm::StringRef S3("3");

  auto Hash123andEmpty = computeHash<HE>(S123, SEmpty);
  auto Hash12And3 = computeHash<HE>(S12, S3);
  auto Hash1And23 = computeHash<HE>(S1, S23);
  auto HashEmptyAnd123 = computeHash<HE>(SEmpty, S123);

  EXPECT_NE(Hash123andEmpty, Hash12And3);
  EXPECT_NE(Hash123andEmpty, Hash1And23);
  EXPECT_NE(Hash123andEmpty, HashEmptyAnd123);
  EXPECT_NE(Hash12And3, Hash1And23);
  EXPECT_NE(Hash12And3, HashEmptyAnd123);
  EXPECT_NE(Hash1And23, HashEmptyAnd123);
}

TYPED_TEST(HashBuilderTest, HashStdString) {
  using HE = TypeParam;
  EXPECT_EQ(computeHash<HE>(std::string("123")),
            computeHash<HE>(llvm::StringRef("123")));
}

TYPED_TEST(HashBuilderTest, HashStdPairTuple) {
  using HE = TypeParam;
  EXPECT_EQ(computeHash<HE>(std::make_pair(1, "string")),
            computeHash<HE>(1, "string"));
  EXPECT_EQ(computeHash<HE>(std::make_tuple(1, "string", 3.0f)),
            computeHash<HE>(1, "string", 3.0f));

  std::pair<StructWithoutCopyOrMove, std::string> Pair;
  Pair.first.I = 1;
  Pair.second = "string";
  std::tuple<StructWithoutCopyOrMove, std::string> Tuple;
  std::get<0>(Tuple).I = 1;
  std::get<1>(Tuple) = "string";

  EXPECT_EQ(computeHash<HE>(Pair), computeHash<HE>(Tuple));
}

TYPED_TEST(HashBuilderTest, HashRangeWithForwardIterator) {
  using HE = TypeParam;
  std::list<int> List;
  List.push_back(1);
  List.push_back(2);
  List.push_back(3);
  EXPECT_NE(computeHashForRange<HE>(List), computeHash<HE>(1, 2, 3));
}
