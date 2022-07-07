// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <memory>

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/blink/public/web/web_heap.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/core/frame/navigator.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_typed_array.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

class MLGraphBuilderTest : public testing::Test {
 public:
  MLGraphBuilderTest() = default;
  ~MLGraphBuilderTest() override = default;
};

MLGraphBuilder* CreateMLGraphBuilder(V8TestingScope& v8_scope) {
  auto* ml = ML::ml(*v8_scope.GetWindow().navigator());
  MLContextOptions* context_options = MLContextOptions::Create();
  context_options->setDevicePreference(V8MLDevicePreference::Enum::kAuto);
  context_options->setPowerPreference(V8MLPowerPreference::Enum::kAuto);
  MLContext* context =
      ml->createContextSync(context_options, v8_scope.GetExceptionState());
  MLGraphBuilder* mlGraphBuilder = MLGraphBuilder::Create(context);
  return mlGraphBuilder;
}

MLOperand* CreateInput(
    V8TestingScope& v8_scope,
    MLGraphBuilder* ml_graph_builder,
    const WTF::Vector<int32_t>& input_shape,
    V8MLOperandType::Enum type = V8MLOperandType::Enum::kFloat32) {
  MLOperandDescriptor* ml_operand_desc = MLOperandDescriptor::Create();
  ml_operand_desc->setDimensions(input_shape);
  ml_operand_desc->setType(type);
  MLOperand* input = ml_graph_builder->input("input", ml_operand_desc,
                                           v8_scope.GetExceptionState());
  return input;
}

MLOperand* CreateConstant(
    V8TestingScope& v8_scope,
    MLGraphBuilder* ml_graph_builder,
    const WTF::Vector<int32_t>& shape,
    V8MLOperandType::Enum type = V8MLOperandType::Enum::kFloat32) {
  int32_t size = 1;
  for (auto i : shape) {
    size *= i;
  }
  MaybeShared<DOMArrayBufferView> buffer_view;
  switch (type) {
    case V8MLOperandType::Enum::kFloat32: {
      auto* float32_array = blink::DOMFloat32Array::Create(size);
      for (int32_t i = 0; i < size; i++) {
        float32_array->Data()[i] = 1;
      }
      buffer_view = MaybeShared<DOMArrayBufferView>(float32_array);
      break;
    }
    case V8MLOperandType::Enum::kInt32: {
      auto* int32_array = blink::DOMInt32Array::Create(size);
      for (int32_t i = 0; i < size; i++) {
        int32_array->Data()[i] = 1;
      }
      buffer_view = MaybeShared<DOMArrayBufferView>(int32_array);
      break;
    }
    default:
      NOTREACHED();
  }
  MLOperandDescriptor* operand_desc = MLOperandDescriptor::Create();
  operand_desc->setDimensions(shape);
  operand_desc->setType(type);
  MLOperand* constant = ml_graph_builder->constant(operand_desc, buffer_view,
                                                 v8_scope.GetExceptionState());
  return constant;
}

TEST_F(MLGraphBuilderTest, Conv2dFilterTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  ASSERT_TRUE(v8_scope.GetExceptionState().Code() ==
              ToExceptionCode(DOMExceptionCode::kNoError));
  MLOperand* input = CreateInput(v8_scope, ml_graph_builder, {1, 1, 5, 5});
  ASSERT_TRUE(v8_scope.GetExceptionState().Code() ==
              ToExceptionCode(DOMExceptionCode::kNoError));
  {
    MLOperand* filter = CreateConstant(v8_scope, ml_graph_builder, {1, 2});
    MLConv2dOptions* options = MLConv2dOptions::Create();
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The filter is not a 4-D tensor.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLOperand* filter =
        CreateConstant(v8_scope, ml_graph_builder, {1, 1, 2, 2},
                       V8MLOperandType::Enum::kInt32);
    MLConv2dOptions* options = MLConv2dOptions::Create();
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The filter type is not consistent with input.",
              v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, Conv2dOptionsTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input = CreateInput(v8_scope, ml_graph_builder, {1, 1, 5, 5});
  MLOperand* filter = CreateConstant(v8_scope, ml_graph_builder, {1, 1, 2, 2});
  ASSERT_TRUE(v8_scope.GetExceptionState().Code() ==
              ToExceptionCode(DOMExceptionCode::kNoError));
  {
    MLConv2dOptions* options = blink::MLConv2dOptions::Create();
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_NE(conv, nullptr);
    EXPECT_FALSE(options->hasActivation());
    EXPECT_TRUE(options->hasAutoPad());
    EXPECT_EQ(options->autoPad(), V8MLAutoPad::Enum::kExplicit);
    EXPECT_FALSE(options->hasBias());
    EXPECT_FALSE(options->hasDilations());
    EXPECT_FALSE(options->hasActivation());
    EXPECT_TRUE(options->hasFilterLayout());
    EXPECT_EQ(options->filterLayout(),
              V8MLConv2dFilterOperandLayout::Enum::kOihw);
    EXPECT_TRUE(options->hasInputLayout());
    EXPECT_EQ(options->inputLayout(), V8MLInputOperandLayout::Enum::kNchw);
    EXPECT_TRUE(options->hasGroups());
    EXPECT_EQ(options->groups(), 1);
    EXPECT_FALSE(options->hasPadding());
    EXPECT_FALSE(options->hasStrides());
  }

  {
    MLConv2dOptions* options = MLConv2dOptions::Create();
    options->setPadding({1, 1});
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of padding is not 4.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLConv2dOptions* options = MLConv2dOptions::Create();
    options->setStrides({2});
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of strides is not 2.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLConv2dOptions* options = MLConv2dOptions::Create();
    options->setDilations({1});
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of dilations is not 2.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLConv2dOptions* options = MLConv2dOptions::Create();
    MLOperand* bias = CreateConstant(v8_scope, ml_graph_builder, {1, 2});
    options->setBias(bias);
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The bias is not a 1-D tensor.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLConv2dOptions* options = MLConv2dOptions::Create();
    MLOperand* bias = CreateConstant(v8_scope, ml_graph_builder, {2});
    options->setBias(bias);
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The bias shape is not [output_channels].",
              v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, Conv2dGroupOptionsTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input = CreateInput(v8_scope, ml_graph_builder, {1, 4, 5, 5});
  MLOperand* filter = CreateConstant(v8_scope, ml_graph_builder, {1, 1, 2, 2});
  MLConv2dOptions* options = MLConv2dOptions::Create();
  {
    options->setGroups(4);
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_NE(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kNoError));
  }
  {
    options->setGroups(2);
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ(
        "The groups is invalid, it must evenly divide the input channels to "
        "filter input depth.",
        v8_scope.GetExceptionState().Message());
  }
  {
    options->setGroups(3);
    MLOperand* conv = ml_graph_builder->conv2d(input, filter, options,
                                             v8_scope.GetExceptionState());
    EXPECT_EQ(conv, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ(
        "The groups is invalid, it must evenly divide the input channels to "
        "filter input depth.",
        v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, AveragePool2dInputTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  {
    MLOperand* input = CreateInput(v8_scope, ml_graph_builder, {1, 5, 5});
    MLPool2dOptions* options = MLPool2dOptions::Create();
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The input is not a 4-D tensor.",
              v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, AveragePool2dOptionsTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input = CreateInput(v8_scope, ml_graph_builder, {1, 1, 5, 5});
  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_NE(pool, nullptr);
    EXPECT_FALSE(options->hasWindowDimensions());
    EXPECT_FALSE(options->hasPadding());
    EXPECT_FALSE(options->hasStrides());
    EXPECT_FALSE(options->hasDilations());
    EXPECT_TRUE(options->hasAutoPad());
    EXPECT_EQ(options->autoPad(), V8MLAutoPad::Enum::kExplicit);
    EXPECT_TRUE(options->hasLayout());
    EXPECT_EQ(options->layout(), V8MLInputOperandLayout::Enum::kNchw);
    EXPECT_TRUE(options->hasRoundingType());
    EXPECT_EQ(options->roundingType(), V8MLRoundingType::Enum::kFloor);
    EXPECT_FALSE(options->hasOutputSizes());
  }

  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setWindowDimensions({1, 1, 1});
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of windowDimensions is not 2.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setPadding({1, 1, 1});
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of padding is not 4.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setStrides({2});
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of strides is not 2.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setDilations({2});
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of dilations is not 2.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setOutputSizes({1, 1, 5, 5});
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    EXPECT_EQ(pool, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The length of outputSizes is not 2.",
              v8_scope.GetExceptionState().Message());
  }

  {
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setAutoPad(V8MLAutoPad::Enum::kSameUpper);
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    WTF::Vector<int32_t> output_size = pool->Dimensions();
    EXPECT_EQ(output_size[0], 1);
    EXPECT_EQ(output_size[1], 1);
    EXPECT_EQ(output_size[2], 5);
    EXPECT_EQ(output_size[3], 5);
  }

  {
    input = CreateInput(v8_scope, ml_graph_builder, {1, 1, 7, 7});
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setWindowDimensions({4, 4});
    options->setStrides({2, 2});
    options->setAutoPad(V8MLAutoPad::Enum::kSameLower);
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    WTF::Vector<int32_t> output_size = pool->Dimensions();
    EXPECT_EQ(output_size[0], 1);
    EXPECT_EQ(output_size[1], 1);
    EXPECT_EQ(output_size[2], 4);
    EXPECT_EQ(output_size[3], 4);
  }

  {
    input = CreateInput(v8_scope, ml_graph_builder, {1, 1, 7, 7});
    MLPool2dOptions* options = MLPool2dOptions::Create();
    options->setWindowDimensions({4, 4});
    options->setStrides({2, 2});
    options->setPadding({1, 1, 1, 1});
    options->setRoundingType(V8MLRoundingType::Enum::kCeil);
    MLOperand* pool = ml_graph_builder->averagePool2d(
        input, options, v8_scope.GetExceptionState());
    WTF::Vector<int32_t> output_size = pool->Dimensions();
    EXPECT_EQ(output_size[0], 1);
    EXPECT_EQ(output_size[1], 1);
    EXPECT_EQ(output_size[2], 4);
    EXPECT_EQ(output_size[3], 4);
  }
}

TEST_F(MLGraphBuilderTest, ReshapeTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input = CreateInput(v8_scope, ml_graph_builder, {2, 3, 4});
  {
    const WTF::Vector<int32_t> new_shape = {3, -1, 8};
    MLOperand* reshape =
        ml_graph_builder->reshape(input, new_shape, v8_scope.GetExceptionState());
    EXPECT_NE(reshape, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kNoError));
  }
  {
    const WTF::Vector<int32_t> new_shape = {2, -1, 5};
    MLOperand* reshape =
        ml_graph_builder->reshape(input, new_shape, v8_scope.GetExceptionState());
    EXPECT_EQ(reshape, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The new shape is invalid.",
              v8_scope.GetExceptionState().Message());
  }
  {
    const WTF::Vector<int32_t> new_shape = {2, -1, -1, 6};
    MLOperand* reshape =
        ml_graph_builder->reshape(input, new_shape, v8_scope.GetExceptionState());
    EXPECT_EQ(reshape, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The new shape is invalid.",
              v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, GemmInputsTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* mlGraphBuilder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input_a = CreateInput(v8_scope, mlGraphBuilder, {2, 3});
  {
    MLOperand* input_b = CreateInput(v8_scope, mlGraphBuilder, {3, 4}, V8MLOperandType::Enum::kInt32);
    MLGemmOptions* options = MLGemmOptions::Create();
    MLOperand* gemm = mlGraphBuilder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_EQ(gemm, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The types of input a and b are inconsistent.",
              v8_scope.GetExceptionState().Message());
  }
  {
    MLOperand* input_b = CreateInput(v8_scope, mlGraphBuilder, {2, 3});
    MLGemmOptions* options = MLGemmOptions::Create();
    MLOperand* gemm = mlGraphBuilder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_EQ(gemm, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The shapes of input a and b are invalid for matrix multiplication.",
              v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, GemmOptionsTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input_a = CreateInput(v8_scope, ml_graph_builder, {3, 4});
  MLOperand* input_b = CreateInput(v8_scope, ml_graph_builder, {4, 5});
  {
    MLGemmOptions* options = MLGemmOptions::Create();
    MLOperand* gemm = ml_graph_builder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_NE(gemm, nullptr);
    EXPECT_FALSE(options->hasC());
    EXPECT_TRUE(options->hasAlpha());
    EXPECT_EQ(options->alpha(), 1);
    EXPECT_TRUE(options->hasBeta());
    EXPECT_EQ(options->beta(), 1);
    EXPECT_TRUE(options->hasATranspose());
    EXPECT_EQ(options->aTranspose(), false);
    EXPECT_TRUE(options->hasBTranspose());
    EXPECT_EQ(options->bTranspose(), false);
  }
  {
    MLGemmOptions* options = MLGemmOptions::Create();
    MLOperand* input_c = CreateInput(v8_scope, ml_graph_builder, {});
    options->setC(input_c);
    MLOperand* gemm = ml_graph_builder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_NE(gemm, nullptr);
  }
  {
    MLGemmOptions* options = MLGemmOptions::Create();
    MLOperand* input_c = CreateInput(v8_scope, ml_graph_builder, {1, 5});
    options->setC(input_c);
    MLOperand* gemm = ml_graph_builder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_NE(gemm, nullptr);
  }
  {
    MLGemmOptions* options = MLGemmOptions::Create();
    MLOperand* input_c = CreateInput(v8_scope, ml_graph_builder, {3, 4});
    options->setC(input_c);
    MLOperand* gemm = ml_graph_builder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_EQ(gemm, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The shape of input c is invalid.",
              v8_scope.GetExceptionState().Message());
  }
  {
    MLGemmOptions* options = MLGemmOptions::Create();
    options->setATranspose(true);
    MLOperand* gemm = ml_graph_builder->gemm(input_a, input_b, options,
                                           v8_scope.GetExceptionState());
    EXPECT_EQ(gemm, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ(
        "The shapes of input a and b are invalid for matrix multiplication.",
        v8_scope.GetExceptionState().Message());
  }
}

TEST_F(MLGraphBuilderTest, AddTest) {
  V8TestingScope v8_scope;
  MLGraphBuilder* ml_graph_builder = CreateMLGraphBuilder(v8_scope);
  MLOperand* input_a = CreateInput(v8_scope, ml_graph_builder, {3, 4});
  {
    MLOperand* input_b = CreateInput(v8_scope, ml_graph_builder, {3, 4},
                                     V8MLOperandType::Enum::kInt32);
    MLOperand* add =
        ml_graph_builder->add(input_a, input_b, v8_scope.GetExceptionState());
    EXPECT_EQ(add, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The types of input a and b are inconsistent.",
              v8_scope.GetExceptionState().Message());
  }
  {
    MLOperand* input_b = CreateInput(v8_scope, ml_graph_builder, {3, 2});
    MLOperand* add =
        ml_graph_builder->add(input_a, input_b, v8_scope.GetExceptionState());
    EXPECT_EQ(add, nullptr);
    EXPECT_EQ(v8_scope.GetExceptionState().Code(),
              ToExceptionCode(DOMExceptionCode::kDataError));
    EXPECT_EQ("The shapes of input a and b are not broadcast compatible.",
              v8_scope.GetExceptionState().Message());
  }
}
}  // namespace blink