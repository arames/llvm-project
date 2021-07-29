// RUN: mlir-opt --split-input-file --verify-diagnostics --allow-unregistered-dialect --test-join-meet-type-interface %s | FileCheck %s

#encoding1 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ]
}>

#encoding2 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense" ]
}>

#encoding3 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense" ]
}>

#map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

func @test_join(
           %i8 : i8,
           %i32 : i32,

           %tensor_unranked_i8 : tensor<*xi8>,
           %tensor_unranked_i32 : tensor<*xi32>,

           %tensor_i32 : tensor<i32>,
           %tensor_1xi32 : tensor<1xi32>,
           %tensor_2xi32 : tensor<2xi32>,
           %tensor_3x4xi32 : tensor<3x4xi32>,
           %tensor_5x6x_xi32 : tensor<5x6x?xi32>,
           %tensor_5x_x7xi32 : tensor<5x?x7xi32>,

           %tensor_1xi32_encoding1 : tensor<1xi32, #encoding1>,
           %tensor_2xi32_encoding1 : tensor<2xi32, #encoding1>,
           %tensor_2xi32_encoding2 : tensor<2xi32, #encoding2>,
           %tensor_2x2xi32_encoding3 : tensor<2x2xi32, #encoding3>,

           %memref_unranked_i32_memspace_1 : memref<*xi32, 1>,
           %memref_i32_memspace_1 : memref<i32, 1>,
           %memref_i32_memspace_2 : memref<i32, 2>,
           %memref_1x2x_xi32_map_1 : memref<1x2x?xi32, #map1>,
           %memref_1x_x3xi32_map_1 : memref<1x?x3xi32, #map1>,
           %memref_1x1x1xi32_map_2 : memref<1x1x1xi32, #map2>
           ) {
  // Test identity.

  "join"(%i8, %i8) : (i8, i8) -> i1
  // CHECK: (i8, i8) -> i8

  // Test different types and element types.

  "join"(%i8, %i32) : (i8, i32) -> i1
  // expected-error@-1 {{types do not join}}

  "join"(%tensor_unranked_i8, %tensor_unranked_i32) : (tensor<*xi8>, tensor<*xi32>) -> i1
  // expected-error@-1 {{types do not join}}


  // Test shapes.

  "join"(%tensor_i32, %i32) : (tensor<i32>, i32) -> i1
  // expected-error@-1 {{types do not join}}

  "join"(%tensor_1xi32, %tensor_1xi32) : (tensor<1xi32>, tensor<1xi32>) -> i1
  // CHECK: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  "join"(%tensor_1xi32, %tensor_3x4xi32) : (tensor<1xi32>, tensor<3x4xi32>) -> i1
  // CHECK: (tensor<1xi32>, tensor<3x4xi32>) -> tensor<*xi32>

  "join"(%tensor_5x6x_xi32, %tensor_5x_x7xi32) : (tensor<5x6x?xi32>, tensor<5x?x7xi32>) -> i1
  // CHECK: (tensor<5x6x?xi32>, tensor<5x?x7xi32>) -> tensor<5x?x?xi32>

  "join"(%tensor_unranked_i32, %tensor_5x6x_xi32) : (tensor<*xi32>, tensor<5x6x?xi32>) -> i1
  // CHECK: (tensor<*xi32>, tensor<5x6x?xi32>) -> tensor<*xi32>


  // Test tensor encoding.

  "join"(%tensor_1xi32_encoding1, %tensor_2xi32_encoding1) : (tensor<1xi32, #encoding1>, tensor<2xi32, #encoding1>) -> i1
  // CHECK: (tensor<1xi32, [[ENCODING1:.*]]>, tensor<2xi32, [[ENCODING1]]>) -> tensor<?xi32, [[ENCODING1]]>

  "join"(%tensor_1xi32_encoding1, %tensor_2xi32_encoding2) : (tensor<1xi32, #encoding1>, tensor<2xi32, #encoding2>) -> i1
  // expected-error@-1 {{types do not join}}

  "join"(%tensor_1xi32_encoding1, %tensor_2x2xi32_encoding3) : (tensor<1xi32, #encoding1>, tensor<2x2xi32, #encoding3>) -> i1
  // expected-error@-1 {{types do not join}}

  "join"(%tensor_1xi32_encoding1, %tensor_unranked_i32) : (tensor<1xi32, #encoding1>, tensor<*xi32>) -> i1
  // expected-error@-1 {{types do not join}}


  // Test memref memory space.

  "join"(%memref_i32_memspace_1, %memref_i32_memspace_1) : (memref<i32, 1>, memref<i32, 1>) -> i1
  // CHECK: (memref<i32, 1>, memref<i32, 1>) -> memref<i32, 1>

  "join"(%memref_i32_memspace_1, %memref_i32_memspace_2) : (memref<i32, 1>, memref<i32, 2>) -> i1
  // expected-error@-1 {{types do not join}}

  "join"(%memref_unranked_i32_memspace_1, %memref_i32_memspace_1) : (memref<*xi32, 1>, memref<i32, 1>) -> i1
  // CHECK: (memref<*xi32, 1>, memref<i32, 1>) -> memref<*xi32, 1>

  "join"(%memref_unranked_i32_memspace_1, %memref_i32_memspace_2) : (memref<*xi32, 1>, memref<i32, 2>) -> i1
  // expected-error@-1 {{types do not join}}


  // Test memref affine map.

  "join"(%memref_1x2x_xi32_map_1, %memref_1x_x3xi32_map_1) : (memref<1x2x?xi32, affine_map<(d0, d1, d2) -> (d1, d2, d0)>>, memref<1x?x3xi32, affine_map<(d0, d1, d2) -> (d1, d2, d0)>>) -> i1
  // CHECK: (memref<1x2x?xi32, [[MAP1:.*]]>, memref<1x?x3xi32, [[MAP1]]>) -> memref<1x?x?xi32, [[MAP1]]>

  "join"(%memref_1x2x_xi32_map_1, %memref_1x1x1xi32_map_2) : (memref<1x2x?xi32, #map1>, memref<1x1x1xi32, #map2>) -> i1
  // expected-error@-1 {{types do not join}}

}

func @test_meet(
           %i8 : i8,
           %i32 : i32,

           %tensor_unranked_i8 : tensor<*xi8>,
           %tensor_unranked_i32 : tensor<*xi32>,

           %tensor_i32 : tensor<i32>,
           %tensor_1xi32 : tensor<1xi32>,
           %tensor_2xi32 : tensor<2xi32>,
           %tensor_3x4xi32 : tensor<3x4xi32>,
           %tensor_5x6x_xi32 : tensor<5x6x?xi32>,
           %tensor_5x_x7xi32 : tensor<5x?x7xi32>,

           %tensor_1xi32_encoding1 : tensor<1xi32, #encoding1>,
           %tensor_2xi32_encoding1 : tensor<2xi32, #encoding1>,
           %tensor_2xi32_encoding2 : tensor<2xi32, #encoding2>,
           %tensor_2x2xi32_encoding3 : tensor<2x2xi32, #encoding3>,

           %memref_unranked_i32_memspace_1 : memref<*xi32, 1>,
           %memref_i32_memspace_1 : memref<i32, 1>,
           %memref_i32_memspace_2 : memref<i32, 2>,
           %memref_1x2x_xi32_map_1 : memref<1x2x?xi32, #map1>,
           %memref_1x_x3xi32_map_1 : memref<1x?x3xi32, #map1>,
           %memref_1x1x1xi32_map_2 : memref<1x1x1xi32, #map2>
           ) {
  // Test identity.

  "meet"(%i8, %i8) : (i8, i8) -> i1
  // CHECK: (i8, i8) -> i8

  // Test different types and element types.

  "meet"(%i8, %i32) : (i8, i32) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%tensor_unranked_i8, %tensor_unranked_i32) : (tensor<*xi8>, tensor<*xi32>) -> i1
  // expected-error@-1 {{types do not meet}}


  // Test shapes.

  "meet"(%tensor_i32, %i32) : (tensor<i32>, i32) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%tensor_1xi32, %tensor_1xi32) : (tensor<1xi32>, tensor<1xi32>) -> i1
  // CHECK: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  "meet"(%tensor_1xi32, %tensor_3x4xi32) : (tensor<1xi32>, tensor<3x4xi32>) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%tensor_5x6x_xi32, %tensor_5x_x7xi32) : (tensor<5x6x?xi32>, tensor<5x?x7xi32>) -> i1
  // CHECK: (tensor<5x6x?xi32>, tensor<5x?x7xi32>) -> tensor<5x6x7xi32>

  "meet"(%tensor_unranked_i32, %tensor_5x6x_xi32) : (tensor<*xi32>, tensor<5x6x?xi32>) -> i1
  // CHECK: (tensor<*xi32>, tensor<5x6x?xi32>) -> tensor<5x6x?xi32>


  // Test tensor encoding.

  "meet"(%tensor_1xi32_encoding1, %tensor_2xi32_encoding1) : (tensor<1xi32, #encoding1>, tensor<2xi32, #encoding1>) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%tensor_1xi32_encoding1, %tensor_2xi32_encoding2) : (tensor<1xi32, #encoding1>, tensor<2xi32, #encoding2>) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%tensor_1xi32_encoding1, %tensor_2x2xi32_encoding3) : (tensor<1xi32, #encoding1>, tensor<2x2xi32, #encoding3>) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%tensor_1xi32_encoding1, %tensor_unranked_i32) : (tensor<1xi32, #encoding1>, tensor<*xi32>) -> i1
  // expected-error@-1 {{types do not meet}}


  // Test memref memory space.

  "meet"(%memref_i32_memspace_1, %memref_i32_memspace_1) : (memref<i32, 1>, memref<i32, 1>) -> i1
  // CHECK: (memref<i32, 1>, memref<i32, 1>) -> memref<i32, 1>

  "meet"(%memref_i32_memspace_1, %memref_i32_memspace_2) : (memref<i32, 1>, memref<i32, 2>) -> i1
  // expected-error@-1 {{types do not meet}}

  "meet"(%memref_unranked_i32_memspace_1, %memref_i32_memspace_1) : (memref<*xi32, 1>, memref<i32, 1>) -> i1
  // CHECK: (memref<*xi32, 1>, memref<i32, 1>) -> memref<i32, 1>

  "meet"(%memref_unranked_i32_memspace_1, %memref_i32_memspace_2) : (memref<*xi32, 1>, memref<i32, 2>) -> i1
  // expected-error@-1 {{types do not meet}}


  // Test memref affine map.

  "meet"(%memref_1x2x_xi32_map_1, %memref_1x_x3xi32_map_1) : (memref<1x2x?xi32, affine_map<(d0, d1, d2) -> (d1, d2, d0)>>, memref<1x?x3xi32, affine_map<(d0, d1, d2) -> (d1, d2, d0)>>) -> i1
  // CHECK: (memref<1x2x?xi32, [[MAP1:.*]]>, memref<1x?x3xi32, [[MAP1]]>) -> memref<1x2x3xi32, [[MAP1]]>

  "meet"(%memref_1x2x_xi32_map_1, %memref_1x1x1xi32_map_2) : (memref<1x2x?xi32, #map1>, memref<1x1x1xi32, #map2>) -> i1
  // expected-error@-1 {{types do not meet}}

}

