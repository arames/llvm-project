// RUN: mlir-opt --split-input-file --allow-unregistered-dialect --test-type-inference-adapter %s | FileCheck %s

func @test_adapter_relax(%x : memref<1x?xi32>, %y : memref<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) : (memref<1x?xi32>, memref<?x2xi32>) -> memref<*xi32>
  "test.ti.disallow_input_type_specialization"(%tmp) : (memref<*xi32>) -> ()
  return
}

// CHECK: [[TMP:%.+]] = "test.ti.join_types"{{.+}} : (memref<1x?xi32>, memref<?x2xi32>) -> memref<?x?xi32>
// CHECK: [[RELAXED_TMP:%.+]] = relax_type [[TMP]] : memref<?x?xi32> to memref<*xi32>
// CHECK: "test.ti.disallow_input_type_specialization"([[RELAXED_TMP]]) : (memref<*xi32>) -> ()

// -----

func @test_adapter_relax(%x : tensor<1x?xi32>, %y : tensor<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.ti.disallow_input_type_specialization"(%tmp) : (tensor<*xi32>) -> ()
  return
}

// CHECK: [[TMP:%.+]] = "test.ti.join_types"{{.+}} : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<?x?xi32>
// CHECK: [[RELAXED_TMP:%.+]] = tensor.cast [[TMP]] : tensor<?x?xi32> to tensor<*xi32>
// CHECK: "test.ti.disallow_input_type_specialization"([[RELAXED_TMP]]) : (tensor<*xi32>) -> ()

// -----

func @test_adapter_specialize(%x : memref<1x?xi32>, %y : memref<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) { allowOutputTypeSpecialization = false } : (memref<1x?xi32>, memref<?x2xi32>) -> memref<*xi32>
  "test.ti.allow_input_type_specialization"(%tmp) : (memref<*xi32>) -> ()
  return
}

// CHECK: [[TMP:%.+]] = "test.ti.join_types"({{.+}}) {allowOutputTypeSpecialization = false} : (memref<1x?xi32>, memref<?x2xi32>) -> memref<*xi32>
// CHECK: [[SPECIALIZED_TMP:%.+]] = specialize_type [[TMP]] : memref<*xi32> to memref<?x?xi32>
// CHECK: "test.ti.allow_input_type_specialization"([[SPECIALIZED_TMP]]) : (memref<?x?xi32>) -> ()

// -----

func @test_adapter_specialize(%x : tensor<1x?xi32>, %y : tensor<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) { allowOutputTypeSpecialization = false } : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.ti.allow_input_type_specialization"(%tmp) : (tensor<*xi32>) -> ()
  return
}

// CHECK: [[TMP:%.+]] = "test.ti.join_types"({{.+}}) {allowOutputTypeSpecialization = false} : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
// CHECK: [[SPECIALIZED_TMP:%.+]] = tensor.cast [[TMP]] : tensor<*xi32> to tensor<?x?xi32>
// CHECK: "test.ti.allow_input_type_specialization"([[SPECIALIZED_TMP]]) : (tensor<?x?xi32>) -> ()
