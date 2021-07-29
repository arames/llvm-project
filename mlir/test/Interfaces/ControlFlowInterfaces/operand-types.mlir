// RUN: mlir-opt -allow-unregistered-dialect --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Test BranchOpInterface
//===----------------------------------------------------------------------===//

func @succ_arg_number_mismatch() {
^bb0:
  %values:2 = "getValues"() : () -> (i1, i17)
  "test.br"(%values#1, %values#0)[^bb1] : (i17, i1) -> ()
  // expected-error@-1 {{branch has 2 operands for successor #0, but target block has 1}}

^bb1(%arg1: i17):
  return
}

// -----

func @succ_arg_type_mismatch() {
^bb0:
  %value = "getValue"() : () -> i1
  "test.br"(%value)[^bb1] : (i1) -> ()
  // expected-error@-1 {{type mismatch for bb argument #0 of successor #0}}

^bb1(%arg1: i32):
  return
}

// -----

func @succ_type_fixed_to_dynamic() {
^bb0:
  %0 = "getValue"() : () -> tensor<1xi32>
  "test.br"(%0)[^bb1] : (tensor<1xi32>) -> ()
  // expected-error@-1 {{type mismatch for bb argument #0 of successor #0}}
^bb1(%arg1: tensor<?xi32>):
  return

^bb2:
  "test.ti.br"(%0)[^bb3] : (tensor<1xi32>) -> ()
^bb3(%arg2: tensor<?xi32>):
  return
}

// -----

func @succ_type_dynamic_to_fixed() {
^bb0:
  %0 = "getValue"() : () -> tensor<?xi32>
  "test.ti.br"(%0)[^bb1] : (tensor<?xi32>) -> ()
  // expected-error@-1 {{type mismatch for bb argument #0 of successor #0}}
^bb1(%arg3: tensor<1xi32>):
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test RegionBranchOpInterface
//===----------------------------------------------------------------------===//

func @succ_arg_type_match() {
  %0 = "getValue"() : () -> (i32)

  %1 = test.region_if %0 : i32 -> (i32) then {
    ^bb0(%arg1 : i32):
      test.region_if_yield %arg1 : i32
  } else {
    ^bb0(%arg1 : i32):
      test.region_if_yield %arg1 : i32
  } join {
    ^bb0(%arg1 : i32):
      test.region_if_yield %arg1 : i32
  }

  return
}

// -----

func @succ_type_fixed_to_dynamic() {
  %0 = "getValue"() : () -> (memref<1xi32>)

  // expected-error@+1 {{'test.region_if' op  along control flow edge from parent operands to Region #1: source type #0 'memref<1xi32>' should match input type #0 'memref<?xi32>'}}
  %tmp1 = test.region_if %0 : memref<1xi32> -> (memref<?xi32>) then {
    ^bb0(%arg1 : memref<1xi32>):
      %true_value = "getValue"(%arg1) : (memref<1xi32>) -> (memref<2xi32>)
      test.region_if_yield %true_value : memref<2xi32>
  } else {
    ^bb0(%arg1 : memref<?xi32>):
      %false_value = "getValue"(%arg1) : (memref<?xi32>) -> (memref<3xi32>)
      test.region_if_yield %false_value : memref<3xi32>
  } join {
    ^bb0(%arg1 : memref<?xi32>):
      test.region_if_yield %arg1 : memref<?xi32>
  }

  %tmp2 = test.ti.region_if %0 : memref<1xi32> -> (memref<?xi32>) then {
    ^bb0(%arg1 : memref<1xi32>):
      %true_value = "getValue"(%arg1) : (memref<1xi32>) -> (memref<2xi32>)
      test.ti.region_if_yield %true_value : memref<2xi32>
  } else {
    ^bb0(%arg1 : memref<?xi32>):
      %false_value = "getValue"(%arg1) : (memref<?xi32>) -> (memref<3xi32>)
      test.ti.region_if_yield %false_value : memref<3xi32>
  } join {
    ^bb0(%arg1 : memref<?xi32>):
      test.ti.region_if_yield %arg1 : memref<?xi32>
  }

  return
}
