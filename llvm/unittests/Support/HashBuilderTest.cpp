//===- llvm/unittest/Support/HashBuilderTest.cpp - HashBuilder unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/HashBuilder.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/SHA1.h"
#include "gtest/gtest.h"

#include <string>

using namespace llvm;

using MD5Words = std::pair<uint64_t, uint64_t>;

// A helper to concisely compute a hash for variadic arguments.
template <typename... Ts> static MD5Words computeMD5Words(const Ts &...Args) {
  MD5 MD5Hash;
  HashBuilder<MD5> HBuilder(MD5Hash);

  HBuilder.update(Args...);

  MD5::MD5Result MD5Res;
  HBuilder.Hasher.final(MD5Res);
  return MD5Res.words();
}

// A helper to concisely compute a hash for a range.
template <typename InputIteratorT>
static MD5Words computeMD5WordsForRange(InputIteratorT First,
                                        InputIteratorT Last) {
  MD5 MD5Hash;
  HashBuilder<MD5> HBuilder(MD5Hash);

  HBuilder.updateRange(First, Last);

  MD5::MD5Result MD5Res;
  HBuilder.Hasher.final(MD5Res);
  return MD5Res.words();
}

enum TestEnumeration { TE_One = 1, TE_Two = 2 };

TEST(HashBuilderTest, BasicTestMD5) {
  char c = 'c';
  int i = 1;
  uint64_t ui64 = static_cast<uint64_t>(1) << 50;
  volatile int vi = 71;
  const volatile int cvi = 71;
  double d = 123.0;

  MD5 MD5Hash;
  HashBuilder<MD5> HBuilder(MD5Hash);

  HBuilder.update(c);
  HBuilder.update(i);
  HBuilder.update(ui64);
  HBuilder.update(TE_One);
  HBuilder.update(vi);
  HBuilder.update(cvi);
  HBuilder.update(d);

  MD5::MD5Result MD5Res;
  HBuilder.Hasher.final(MD5Res);
  EXPECT_EQ(MD5Res.words(),
            MD5Words(0xD2D71BACA77CBDAAULL, 0x6189265BC530E402ULL));
}

TEST(HashBuilderTest, BasicTestSHA1) {
  char c = 'c';
  int i = 1;
  uint64_t ui64 = static_cast<uint64_t>(1) << 50;
  volatile int vi = 71;
  const volatile int cvi = 71;
  double d = 123.0;

  SHA1 SHA1Hash;
  HashBuilder<SHA1> HBuilder(SHA1Hash);

  HBuilder.update(c);
  HBuilder.update(i);
  HBuilder.update(ui64);
  HBuilder.update(TE_One);
  HBuilder.update(vi);
  HBuilder.update(cvi);
  HBuilder.update(d);

  EXPECT_EQ(SHA1Hash.final(),
            "OYR`\x1F\xC6\xF\x93Z\x97o^\xD8\xECh=5\x7F\x9E" "7");
}

struct SimpleStruct {
  char c;
  int i;
};

template <typename HasherT>
void updateHash(HashBuilder<HasherT> &HBuilder, const SimpleStruct &Value) {
  HBuilder.update(Value.c);
  HBuilder.update(Value.i);
}

struct StructWithPrivateMember {
public:
  explicit StructWithPrivateMember(std::string s, float f) : s(s), f(f) {}

  std::string s;

private:
  float f;

  template <typename HasherT>
  friend void updateHash(HashBuilder<HasherT> &HBuilder,
                         const StructWithPrivateMember &Value) {
    HBuilder.update(Value.s);
    HBuilder.update(Value.f);
  }
};

TEST(HashBuilderTest, HashUserDefinedStruct) {
  EXPECT_EQ(computeMD5Words(SimpleStruct{'c', 123}), computeMD5Words('c', 123));
  EXPECT_EQ(computeMD5Words(StructWithPrivateMember{"s", 2.0f}),
            computeMD5Words("s", 2.0f));
}

struct HashableStruct {
  uint16_t i1;
  uint16_t i2;
};
template <>
struct llvm::hash_builder::detail::is_hashable_data<HashableStruct>
    : std::integral_constant<bool, true> {};

TEST(HashBuilderTest, HashHashableStruct) {
  EXPECT_EQ(computeMD5Words(HashableStruct{0x5678, 0x1234}),
            computeMD5Words(0x12345678));
}

TEST(HashBuilderTest, HashArrayRefHashableDataTypes) {
  {
    ArrayRef<int> array{1, 20, 0x12345678};
    EXPECT_EQ(computeMD5Words(array),
              MD5Words(0xA225DC4682CDED92ULL, 0xDC78C8DD549082C0ULL));
    EXPECT_EQ(computeMD5Words(array),
              computeMD5WordsForRange(array.begin(), array.end()));
    EXPECT_EQ(
        computeMD5Words(array),
        computeMD5WordsForRange(array.data(), array.data() + array.size()));
  }

  {
    ArrayRef<HashableStruct> array{{0xa, 0xb}, {0x8001, 0x8002}};
    EXPECT_EQ(computeMD5Words(array),
              MD5Words(0x72756E3DAC25841AULL, 0x19698D20D81A0A37ULL));
    EXPECT_EQ(computeMD5Words(array),
              computeMD5WordsForRange(array.begin(), array.end()));
    EXPECT_EQ(
        computeMD5Words(array),
        computeMD5WordsForRange(array.data(), array.data() + array.size()));
  }
}

TEST(HashBuilderTest, HashArrayRefOrder) {
  ArrayRef<uint8_t> array_123{1, 2, 3};
  ArrayRef<uint8_t> array_empty{};

  ArrayRef<uint8_t> array_12{1, 2};
  ArrayRef<uint8_t> array_3{3};

  ArrayRef<uint8_t> array_1{1};
  ArrayRef<uint8_t> array_23{2, 3};

  MD5Words H_123_E = computeMD5Words(array_123, array_empty);
  MD5Words H_12_3 = computeMD5Words(array_12, array_3);
  MD5Words H_1_23 = computeMD5Words(array_1, array_23);
  MD5Words H_E_123 = computeMD5Words(array_empty, array_123);

  EXPECT_NE(H_123_E, H_12_3);
  EXPECT_NE(H_123_E, H_1_23);
  EXPECT_NE(H_123_E, H_E_123);
  EXPECT_NE(H_12_3, H_1_23);
  EXPECT_NE(H_12_3, H_E_123);
  EXPECT_NE(H_1_23, H_E_123);
}

TEST(HashBuilderTest, HashArrayRefNonHashableDataTypes) {
  ArrayRef<SimpleStruct> array{{'a', 100}, {'b', 200}};
  EXPECT_NE(computeMD5Words(array),
            computeMD5Words(SimpleStruct{'a', 100}, SimpleStruct{'b', 200}));
  EXPECT_EQ(computeMD5Words(array),
            MD5Words(0x2B91D8214B4D2019ULL, 0x9D4F0AE8DC93B382ULL));
}

TEST(HashBuilderTest, HashStringRef) {
  StringRef s123("123");
  StringRef s("");

  StringRef s12("12");
  StringRef s3("3");

  StringRef s1("1");
  StringRef s23("23");

  MD5Words H_123_E = computeMD5Words(s123, s);
  MD5Words H_12_3 = computeMD5Words(s12, s3);
  MD5Words H_1_23 = computeMD5Words(s1, s23);
  MD5Words H_E_123 = computeMD5Words(s, 123);

  EXPECT_NE(H_123_E, H_12_3);
  EXPECT_NE(H_123_E, H_1_23);
  EXPECT_NE(H_123_E, H_E_123);
  EXPECT_NE(H_12_3, H_1_23);
  EXPECT_NE(H_12_3, H_E_123);
  EXPECT_NE(H_1_23, H_E_123);

  EXPECT_EQ(H_123_E, MD5Words(0x40B7CC7F911BBAC6, 0xD36F0009828C85B9));
  EXPECT_EQ(H_12_3, MD5Words(0x2ED6EB15C3A81C34, 0x9C6548CCD926C906));
  EXPECT_EQ(H_1_23, MD5Words(0x54E97BD0C04BD13D, 0x23068458E6DDF5A9));
  EXPECT_EQ(H_E_123, MD5Words(0x6D7068231E0B1AAE, 0x1277BF734D4106A0));
}

TEST(HashBuilderTest, HashStdString) {
  EXPECT_EQ(computeMD5Words(std::string("123")),
            computeMD5Words(StringRef("123")));
}

TEST(HashBuilderTest, HashStdPairTuple) {
  EXPECT_EQ(computeMD5Words(std::make_pair(1, "string")),
            computeMD5Words(std::make_tuple(1, "string")));
}

TEST(HashBuilderTest, HashVariadic) {
  MD5Words VariadicHash;
  MD5Words SerialHash;

  {
    MD5 MD5Hash;
    HashBuilder<MD5> HBuilder(MD5Hash);

    HBuilder.update(100);
    HBuilder.update(2.7);
    HBuilder.update("string");

    MD5::MD5Result MD5Res;
    HBuilder.Hasher.final(MD5Res);
    SerialHash = MD5Res.words();
  }

  {
    MD5 MD5Hash;
    HashBuilder<MD5> HBuilder(MD5Hash);

    HBuilder.update(100, 2.7, "string");

    MD5::MD5Result MD5Res;
    HBuilder.Hasher.final(MD5Res);
    VariadicHash = MD5Res.words();
  }

  EXPECT_EQ(VariadicHash, SerialHash);
}
