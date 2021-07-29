// RUN: mlir-opt -allow-unregistered-dialect -infer-types %s -split-input-file -o %t1
// RUN: FileCheck %s < %t1
// Verify idempotence.
// RUN: mlir-opt -allow-unregistered-dialect -infer-types %t1 -split-input-file -o %t2
// RUN: FileCheck %s < %t2

func @test_input_specialization(%x : tensor<1x?xi32>, %y : tensor<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.ti.disallow_input_type_specialization"(%tmp) : (tensor<*xi32>) -> ()
  "test.ti.allow_input_type_specialization"(%tmp) : (tensor<*xi32>) -> ()
  "test.ti.conditionally_allow_input_type_specialization"(%tmp, %tmp) : (tensor<*xi32>, tensor<*xi32>) -> ()
  return
}

// CHECK: [[TMP:%.+]] = "test.ti.join_types"{{.+}} : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<?x?xi32>
// CHECK: [[RELAXED_TMP:%.+]] = relax_type [[TMP]] : tensor<?x?xi32> to tensor<*xi32>
// CHECK: "test.ti.disallow_input_type_specialization"([[RELAXED_TMP]]) : (tensor<*xi32>) -> ()
// CHECK: "test.ti.allow_input_type_specialization"([[TMP]]) : (tensor<?x?xi32>) -> ()
// CHECK: "test.ti.conditionally_allow_input_type_specialization"([[TMP]], [[RELAXED_TMP]]) : (tensor<?x?xi32>, tensor<*xi32>) -> ()

// -----

func @test_output_specialization(%x : tensor<1x?xi32>, %y : tensor<?x2xi32>) {
  %join1 = "test.ti.join_types"(%x, %y) { allowOutputTypeSpecialization = false } : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.ti.allow_input_type_specialization"(%join1) : (tensor<*xi32>) -> ()

  %join2 = "test.ti.join_types"(%x, %y) { allowOutputTypeSpecialization = true } : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.ti.allow_input_type_specialization"(%join2) : (tensor<*xi32>) -> ()
  return
}

// CHECK: [[JOIN1:%.+]] = "test.ti.join_types"({{.+}}) {allowOutputTypeSpecialization = false} : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
// CHECK: [[SPECIALIZED_JOIN1:%.+]] = specialize_type [[JOIN1]] : tensor<*xi32> to tensor<?x?xi32>
// CHECK: "test.ti.allow_input_type_specialization"([[SPECIALIZED_JOIN1]]) : (tensor<?x?xi32>) -> ()
// CHECK: [[JOIN2:%.+]] = "test.ti.join_types"({{.+}}) {allowOutputTypeSpecialization = true} : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<?x?xi32>
// CHECK: "test.ti.allow_input_type_specialization"([[JOIN2]]) : (tensor<?x?xi32>) -> ()

// -----

func @test_relaxation_propagation(%value : tensor<1xi32>) {
  %relaxed_value = relax_type %value : tensor<1xi32> to tensor<*xi32>
  "test.br"(%relaxed_value)[^bb1] : (tensor<*xi32>) -> ()
^bb1(%arg: tensor<*xi32>):
  "test.ti.allow_input_type_specialization"(%arg) : (tensor<*xi32>) -> ()
  return
}

// CHECK: func @test_relaxation_propagation([[VALUE:%.+]]: tensor<1xi32>) {
// CHECK:   [[RELAXED_VALUE:%.+]] = relax_type [[VALUE]] : tensor<1xi32> to tensor<*xi32>
// CHECK:   "test.br"([[RELAXED_VALUE]])[^bb1] : (tensor<*xi32>) -> ()
// CHECK: ^bb1([[ARG:%.+]]: tensor<*xi32>):  // pred: ^bb0
// CHECK:   [[SPECIALIZED_ARG:%.+]] = specialize_type [[ARG]] : tensor<*xi32> to tensor<1xi32>
// CHECK:   "test.ti.allow_input_type_specialization"([[SPECIALIZED_ARG]]) : (tensor<1xi32>) -> ()
// CHECK:   return
// CHECK: }

// -----

func @test_branch_disallowing_input_specialization(%x : tensor<1x?xi32>, %y : tensor<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.br"(%tmp)[^bb1] : (tensor<*xi32>) -> ()
^bb1(%arg: tensor<*xi32>):
  "test.ti.disallow_input_type_specialization"(%arg) : (tensor<*xi32>) -> ()
  "test.ti.allow_input_type_specialization"(%arg) : (tensor<*xi32>) -> ()
  "test.ti.conditionally_allow_input_type_specialization"(%arg, %arg) : (tensor<*xi32>, tensor<*xi32>) -> ()
  return
}

// CHECK:   [[TMP:%.+]] = "test.ti.join_types"({{.+}}) : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<?x?xi32>
// CHECK:   [[RELAXED_TMP:%.+]] = relax_type [[TMP]] : tensor<?x?xi32> to tensor<*xi32>
// CHECK:   "test.br"([[RELAXED_TMP]])[^bb1] : (tensor<*xi32>) -> ()
// CHECK: ^bb1([[ARG:%.+]]: tensor<*xi32>):  // pred: ^bb0
// CHECK:   [[SPECIALIZED_ARG:%.+]] = specialize_type [[ARG]] : tensor<*xi32> to tensor<?x?xi32>
// CHECK:   "test.ti.disallow_input_type_specialization"([[ARG]]) : (tensor<*xi32>) -> ()
// CHECK:   "test.ti.allow_input_type_specialization"([[SPECIALIZED_ARG]]) : (tensor<?x?xi32>) -> ()
// CHECK:   "test.ti.conditionally_allow_input_type_specialization"([[SPECIALIZED_ARG]], [[ARG]]) : (tensor<?x?xi32>, tensor<*xi32>) -> ()
// CHECK:   return

// -----

func @test_branch_allowing_input_specialization(%x : tensor<1x?xi32>, %y : tensor<?x2xi32>) {
  %tmp = "test.ti.join_types"(%x, %y) : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<*xi32>
  "test.ti.br"(%tmp)[^bb1] : (tensor<*xi32>) -> ()
^bb1(%arg: tensor<*xi32>):
  "test.ti.disallow_input_type_specialization"(%arg) : (tensor<*xi32>) -> ()
  "test.ti.allow_input_type_specialization"(%arg) : (tensor<*xi32>) -> ()
  "test.ti.conditionally_allow_input_type_specialization"(%arg, %arg) : (tensor<*xi32>, tensor<*xi32>) -> ()
  return
}

// CHECK:   [[TMP:%.+]] = "test.ti.join_types"({{.+}}) : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<?x?xi32>
// CHECK:   "test.ti.br"([[TMP]])[^bb1] : (tensor<?x?xi32>) -> ()
// CHECK: ^bb1([[ARG:%.+]]: tensor<*xi32>):  // pred: ^bb0
// CHECK:   [[SPECIALIZED_ARG:%.+]] = specialize_type [[ARG]] : tensor<*xi32> to tensor<?x?xi32>
// CHECK:   "test.ti.disallow_input_type_specialization"([[ARG]]) : (tensor<*xi32>) -> ()
// CHECK:   "test.ti.allow_input_type_specialization"([[SPECIALIZED_ARG]]) : (tensor<?x?xi32>) -> ()
// CHECK:   "test.ti.conditionally_allow_input_type_specialization"([[SPECIALIZED_ARG]], [[ARG]]) : (tensor<?x?xi32>, tensor<*xi32>) -> ()
// CHECK:   return

// -----

func @test_branch_join(%cond: i1, %x : tensor<1xi32>, %y : tensor<2xi32>) {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  %val_true = "test.ti.join_types"(%x) : (tensor<1xi32>) -> tensor<*xi32>
  br ^bb3(%val_true : tensor<*xi32>)
^bb2:
  %val_false = "test.ti.join_types"(%y) : (tensor<2xi32>) -> tensor<*xi32>
  br ^bb3(%val_false : tensor<*xi32>)
^bb3(%val: tensor<*xi32>):
  "test.ti.allow_input_type_specialization"(%val) : (tensor<*xi32>) -> ()
  "test.ti.disallow_input_type_specialization"(%val) : (tensor<*xi32>) -> ()
  return
}

// CHECK: func @test_branch_join([[COND:%.+]]: i1, [[X:%.+]]: tensor<1xi32>, [[Y:%.+]]: tensor<2xi32>) {
// CHECK:   cond_br [[COND]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   [[JOIN:%.+]] = "test.ti.join_types"([[X]]) : (tensor<1xi32>) -> tensor<1xi32>
// CHECK:   [[RELAXED_JOIN:%.+]] = relax_type [[JOIN]] : tensor<1xi32> to tensor<*xi32>
// CHECK:   br ^bb3([[RELAXED_JOIN]] : tensor<*xi32>)
// CHECK: ^bb2:
// CHECK:   [[JOIN:%.+]] = "test.ti.join_types"([[Y]]) : (tensor<2xi32>) -> tensor<2xi32>
// CHECK:   [[RELAXED_JOIN:%.+]] = relax_type [[JOIN]] : tensor<2xi32> to tensor<*xi32>
// CHECK:   br ^bb3([[RELAXED_JOIN]] : tensor<*xi32>)
// CHECK: ^bb3([[VAL:%.+]]: tensor<*xi32>):
// CHECK:   [[SPECIALIZED_VAL:%.+]] = specialize_type [[VAL]] : tensor<*xi32> to tensor<?xi32>
// CHECK:   "test.ti.allow_input_type_specialization"([[SPECIALIZED_VAL]]) : (tensor<?xi32>) -> ()
// CHECK:   "test.ti.disallow_input_type_specialization"([[VAL]]) : (tensor<*xi32>) -> ()
// CHECK:   return
// CHECK: }
// -----

func @test_regionbranchopinterface() -> tensor<?x?xi32> {
  %cond = "getValue"() : () -> (i1)

  %res = test.region_if %cond : i1 -> (tensor<?x?xi32>) then {
    ^bb0(%arg1 : i1):
      %tmp1 = "getValue"() : () -> (tensor<1x2xi32>)
      %then_value = "test.ti.join_types"(%tmp1) : (tensor<1x2xi32>) -> tensor<?x?xi32>
      test.region_if_yield %then_value : tensor<?x?xi32>
  } else {
    ^bb0(%arg1 : i1):
      %tmp1 = "getValue"() : () -> (tensor<1x9xi32>)
      %else_value = "test.ti.join_types"(%tmp1) : (tensor<1x9xi32>) -> tensor<?x?xi32>
      test.region_if_yield %else_value : tensor<?x?xi32>
  } join {
    ^bb0(%arg1 : tensor<?x?xi32>):
      test.region_if_yield %arg1 : tensor<?x?xi32>
  }
  "test.ti.allow_input_type_specialization"(%res) : (tensor<?x?xi32>) -> ()
  return %res : tensor<?x?xi32>
}

// CHECK: [[COND:%.+]] = "getValue"() : () -> i1
// CHECK: [[RES:%.+]] = test.region_if [[COND]]: i1 -> tensor<?x?xi32> then {
// CHECK: ^bb0({{.+}}: i1):  // no predecessors
// CHECK:   [[TMP1:%.+]] = "getValue"() : () -> tensor<1x2xi32>
// CHECK:   [[THEN_VALUE:%.+]] = "test.ti.join_types"([[TMP1]]) : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:   [[RELAXED_THEN_VALUE:%.+]] = relax_type [[THEN_VALUE]] : tensor<1x2xi32> to tensor<?x?xi32>
// CHECK:   test.region_if_yield [[RELAXED_THEN_VALUE]] : tensor<?x?xi32>
// CHECK: } else {
// CHECK: ^bb0({{.+}}: i1):  // no predecessors
// CHECK:   [[TMP2:%.+]] = "getValue"() : () -> tensor<1x9xi32>
// CHECK:   [[ELSE_VALUE:%.+]] = "test.ti.join_types"([[TMP2]]) : (tensor<1x9xi32>) -> tensor<1x9xi32>
// CHECK:   [[RELAXED_ELSE_VALUE:%.+]] = relax_type [[ELSE_VALUE]] : tensor<1x9xi32> to tensor<?x?xi32>
// CHECK:   test.region_if_yield [[RELAXED_ELSE_VALUE]] : tensor<?x?xi32>
// CHECK: } join {
// CHECK: ^bb0([[ARG:%.+]]: tensor<?x?xi32>):  // no predecessors
// CHECK:   test.region_if_yield [[ARG]] : tensor<?x?xi32>
// CHECK: }
// CHECK: [[SPECIALIZED_RES:%.+]] = specialize_type [[RES]] : tensor<?x?xi32> to tensor<1x?xi32>
// CHECK: "test.ti.allow_input_type_specialization"([[SPECIALIZED_RES]]) : (tensor<1x?xi32>) -> ()
// CHECK: return [[RES]] : tensor<?x?xi32>

// -----

func @test_regionbranchopinterface() -> tensor<?x?xi32> {
  %cond = "getValue"() : () -> (i1)

  %res = test.ti.region_if %cond : i1 -> (tensor<?x?xi32>) then {
    ^bb0(%arg1 : i1):
      %tmp1 = "getValue"() : () -> (tensor<1x2xi32>)
      %then_value = "test.ti.join_types"(%tmp1) : (tensor<1x2xi32>) -> tensor<?x?xi32>
      test.ti.region_if_yield %then_value : tensor<?x?xi32>
  } else {
    ^bb0(%arg1 : i1):
      %tmp1 = "getValue"() : () -> (tensor<1x9xi32>)
      %else_value = "test.ti.join_types"(%tmp1) : (tensor<1x9xi32>) -> tensor<?x?xi32>
      test.ti.region_if_yield %else_value : tensor<?x?xi32>
  } join {
    ^bb0(%arg1 : tensor<?x?xi32>):
      test.ti.region_if_yield %arg1 : tensor<?x?xi32>
  }
  "test.ti.allow_input_type_specialization"(%res) : (tensor<?x?xi32>) -> ()
  return %res : tensor<?x?xi32>
}

// CHECK: [[COND:%.+]] = "getValue"() : () -> i1
// CHECK: [[RES:%.+]] = test.ti.region_if [[COND]]: i1 -> tensor<1x?xi32> then {
// CHECK: ^bb0({{.+}}: i1):  // no predecessors
// CHECK:   [[TMP1:%.+]] = "getValue"() : () -> tensor<1x2xi32>
// CHECK:   [[THEN_VALUE:%.+]] = "test.ti.join_types"([[TMP1]]) : (tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:   test.ti.region_if_yield [[THEN_VALUE]] : tensor<1x2xi32>
// CHECK: } else {
// CHECK: ^bb0({{.+}}: i1):  // no predecessors
// CHECK:   [[TMP2:%.+]] = "getValue"() : () -> tensor<1x9xi32>
// CHECK:   [[ELSE_VALUE:%.+]] = "test.ti.join_types"([[TMP2]]) : (tensor<1x9xi32>) -> tensor<1x9xi32>
// CHECK:   test.ti.region_if_yield [[ELSE_VALUE]] : tensor<1x9xi32>
// CHECK: } join {
// CHECK: ^bb0([[ARG:%.+]]: tensor<?x?xi32>):  // no predecessors
// CHECK:   [[SPECIALIZED_ARG:%.+]] = specialize_type [[ARG]] : tensor<?x?xi32> to tensor<1x?xi32>
// CHECK:   test.ti.region_if_yield [[SPECIALIZED_ARG]] : tensor<1x?xi32>
// CHECK: }
// CHECK: [[RELAXED_RES:%.+]] = relax_type [[RES]] : tensor<1x?xi32> to tensor<?x?xi32>
// CHECK: "test.ti.allow_input_type_specialization"([[RES]]) : (tensor<1x?xi32>) -> ()
// CHECK: return [[RELAXED_RES]] : tensor<?x?xi32>

// -----

func @test_dynamicize(%arg : tensor<1x2x3xi32>) {
  %tmp1 = "test.ti.dynamicize"(%arg) : (tensor<1x2x3xi32>) -> tensor<*xi32>
  %tmp2 = "test.ti.dynamicize"(%tmp1) : (tensor<*xi32>) -> tensor<*xi32>
  %tmp3 = "test.ti.dynamicize"(%tmp2) : (tensor<*xi32>) -> tensor<*xi32>
  %tmp4 = "test.ti.dynamicize"(%tmp3) : (tensor<*xi32>) -> tensor<*xi32>
  return
}

// CHECK: func @test_dynamicize([[ARG:%.+]]: tensor<1x2x3xi32>) {
// CHECK: [[TMP1:%.+]] = "test.ti.dynamicize"([[ARG]]) : (tensor<1x2x3xi32>) -> tensor<?x2x3xi32>
// CHECK: [[TMP2:%.+]] = "test.ti.dynamicize"([[TMP1]]) : (tensor<?x2x3xi32>) -> tensor<?x?x3xi32>
// CHECK: [[TMP3:%.+]] = "test.ti.dynamicize"([[TMP2]]) : (tensor<?x?x3xi32>) -> tensor<?x?x?xi32>
// CHECK: [[TMP4:%.+]] = "test.ti.dynamicize"([[TMP3]]) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>

// -----

func @test_simple_while(%func_arg : tensor<1x2x3xi32>) {
  %relaxed_func_arg = relax_type %func_arg : tensor<1x2x3xi32> to tensor<*xi32>
  %while = scf.while (%before_arg = %relaxed_func_arg) : (tensor<*xi32>) -> tensor<*xi32> {
    %cond = "getValue"() : () -> (i1)
    scf.condition(%cond) %before_arg : tensor<*xi32>
  } do {
  ^bb0(%after_arg: tensor<*xi32>):
    %join = "test.ti.join_types"(%after_arg) : (tensor<*xi32>) -> (tensor<*xi32>)
    scf.yield %join : tensor<*xi32>
  }
  "test.ti.allow_input_type_specialization"(%while) : (tensor<*xi32>) -> ()
  return
}

// CHECK: func @test_simple_while([[FUNC_ARG:%.+]]: tensor<1x2x3xi32>) {
// CHECK:   [[RELAXED_FUNC_ARG:%.+]] = relax_type [[FUNC_ARG]] : tensor<1x2x3xi32> to tensor<*xi32>
// CHECK:   [[WHILE:%.+]] = scf.while ([[BEFORE_ARG:%.+]] = [[RELAXED_FUNC_ARG]]) : (tensor<*xi32>) -> tensor<*xi32> {
// CHECK:     [[COND:%.+]] = "getValue"() : () -> i1
// CHECK:     scf.condition([[COND]]) [[BEFORE_ARG]] : tensor<*xi32>
// CHECK:   } do {
// CHECK:   ^bb0([[AFTER_ARG:%.+]]: tensor<*xi32>):  // no predecessors
// CHECK:     [[JOIN:%.+]] = "test.ti.join_types"([[AFTER_ARG]]) : (tensor<*xi32>) -> tensor<1x2x3xi32>
// CHECK:     [[RELAXED_JOIN:%.+]] = relax_type [[JOIN]] : tensor<1x2x3xi32> to tensor<*xi32>
// CHECK:     scf.yield [[RELAXED_JOIN]] : tensor<*xi32>
// CHECK:   }
// CHECK:   [[SPECIALIZED_WHILE:%.+]] = specialize_type [[WHILE]] : tensor<*xi32> to tensor<1x2x3xi32>
// CHECK:   "test.ti.allow_input_type_specialization"([[SPECIALIZED_WHILE]]) : (tensor<1x2x3xi32>) -> ()
// CHECK:   return
// CHECK: }

// -----

func @test_while(%func_arg : tensor<1x2x3xi32>) {
  %relaxed_func_arg = relax_type %func_arg : tensor<1x2x3xi32> to tensor<*xi32>
  %while = scf.while (%before_arg = %relaxed_func_arg) : (tensor<*xi32>) -> tensor<*xi32> {
    %cond = "getValue"() : () -> (i1)
    scf.condition(%cond) %before_arg : tensor<*xi32>
  } do {
  ^bb0(%after_arg: tensor<*xi32>):
    %dyn = "test.ti.dynamicize"(%after_arg) : (tensor<*xi32>) -> (tensor<*xi32>)
    scf.yield %dyn : tensor<*xi32>
  }
  "test.ti.allow_input_type_specialization"(%while) : (tensor<*xi32>) -> ()
  return
}

// CHECK: func @test_while([[FUNC_ARG:%.+]]: tensor<1x2x3xi32>) {
// CHECK:   [[RELAXED_FUNC_ARG:%.+]] = relax_type [[FUNC_ARG]] : tensor<1x2x3xi32> to tensor<*xi32>
// CHECK:   [[WHILE:%.+]] = scf.while ([[BEFORE_ARG:%.+]] = [[RELAXED_FUNC_ARG]]) : (tensor<*xi32>) -> tensor<*xi32> {
// CHECK:     [[COND:%.+]] = "getValue"() : () -> i1
// CHECK:     scf.condition([[COND]]) [[BEFORE_ARG]] : tensor<*xi32>
// CHECK:   } do {
// CHECK:   ^bb0([[AFTER_ARG:%.+]]: tensor<*xi32>):  // no predecessors
// CHECK:     [[SPECIALIZED_AFTER_ARG:%.+]] = specialize_type [[AFTER_ARG]] : tensor<*xi32> to tensor<?x?x?xi32>
// CHECK:     [[DYN:%.+]] = "test.ti.dynamicize"([[SPECIALIZED_AFTER_ARG]]) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK:     [[RELAXED_DYN:%.+]] = relax_type [[DYN]] : tensor<?x?x?xi32> to tensor<*xi32>
// CHECK:     scf.yield [[RELAXED_DYN]] : tensor<*xi32>
// CHECK:   }
// CHECK:   [[SPECIALIZED_WHILE:%.+]] = specialize_type [[WHILE]] : tensor<*xi32> to tensor<?x?x?xi32>
// CHECK:   "test.ti.allow_input_type_specialization"([[SPECIALIZED_WHILE]]) : (tensor<?x?x?xi32>) -> ()
// CHECK:   return
// CHECK: }
