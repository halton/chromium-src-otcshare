// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ie_compilation.h"

#include <string>
#include <utility>

#include "constants.h"
#include "utils.h"

#include "ngraph/ngraph.hpp"
namespace InferenceEngine {
using namespace ngraph;
namespace {

int32_t ConvertDims(const std::vector<uint32_t>& dimensions,
                    std::vector<size_t>& dims) {
  // IE default layout is NCHW
  dims.resize(dimensions.size());
  if (dimensions.size() == 1) {
    dims[0] = dimensions[0];
  } else if (dimensions.size() == 2) {
    dims[0] = dimensions[0];
    dims[1] = dimensions[1];
  } else if (dimensions.size() == 3) {
    // HWC -> CHW
    dims[0] = dimensions[2];
    dims[1] = dimensions[0];
    dims[2] = dimensions[1];
  } else if (dimensions.size() == 4) {
    // NHWC -> NCHW
    dims[0] = dimensions[0];
    dims[1] = dimensions[3];
    dims[2] = dimensions[1];
    dims[3] = dimensions[2];
  } else {
    std::cout << "Tensor rank " << dimensions.size() << " is not supproted"
              << std::endl;
    return error_t::BAD_DATA;
  }
  return error_t::NOT_ERROR;
}
}  // namespace

Compilation::Compilation(ModelInfoPtr model)
    : model_(model), network_(nullptr) {}

void Compilation::SetPreference(int32_t prefer) {
  preference_ = prefer;
}

int32_t Compilation::GetPreference() {
  return preference_;
}

int32_t Compilation::Compile() {
  int32_t result;
  for (size_t i = 0; i < model_->inputs.size(); ++i) {
    result = AddInput(model_->inputs[i]);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model_->operations.size(); ++i) {
    const Operation& operation = model_->operations[i];
    const int32_t type = operation.type;

    if (operation.outputs.size() != 1) {
      std::cout << "Only 1 output is supported";
      return error_t::BAD_DATA;
    }

    int32_t result = error_t::NOT_ERROR;
    if (type == operation_t::ADD || type == operation_t::MUL) {
      result = AddElementwise(operation);
    } else if (type == operation_t::CONV_2D ||
               type == operation_t::DEPTHWISE_CONV_2D ||
               type == operation_t::ATROUS_CONV_2D ||
               type == operation_t::ATROUS_DEPTHWISE_CONV_2D) {
      result = AddConvolution(operation);
      // } else if (type == operation_t::AVERAGE_POOL_2D ||
      //            type == operation_t::MAX_POOL_2D) {
      //   result = AddPooling(operation);
      // } else if (type == operation_t::SOFTMAX) {
      //   result = AddSoftmax(operation);
    } else if (type == operation_t::RESHAPE) {
      result = AddReshape(operation);
      // } else if (type == operation_t::CONCATENATION) {
      //   result = AddConcatenation(operation);
    } else if (type == operation_t::FULLY_CONNECTED) {
      result = AddFullyConnected(operation);
      // } else if (type == operation_t::RESIZE_BILINEAR_NN) {
      //   result = AddResizeBilinear(operation);
    } else if (type == operation_t::LOGISTIC) {
      result = AddSigmoid(operation);
      // } else if (type == operation_t::ARGMAX) {
      //   result = AddArgmax(operation);
    } else {
      std::cout << "Operation type " << type << " is not supported."
                << std::endl;
      return error_t::BAD_DATA;
    }

    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model_->outputs.size(); ++i) {
    result = AddOutput(model_->outputs[i]);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
  }

  try {
    std::shared_ptr<ngraph::Function> ngraph_function;
    ngraph_function =
        std::make_shared<Function>(ngraph_outputs_, ngraph_inputs_);
    network_.reset(new CNNNetwork(ngraph_function));
    InputsDataMap input_info(network_->getInputsInfo());
    for (auto itr : input_info) {
      if (itr.second->getLayout() == Layout::NCHW) {
        itr.second->setLayout(Layout::NHWC);
      }
      itr.second->setPrecision(Precision::FP32);
    }

    OutputsDataMap output_info(network_->getOutputsInfo());
    for (auto itr : output_info) {
      SizeVector dims = itr.second->getTensorDesc().getDims();
      if (itr.second->getLayout() == Layout::NCHW) {
        itr.second->setLayout(Layout::NHWC);
      }
      itr.second->setPrecision(Precision::FP32);
    }
  } catch (const std::exception& ex) {
    std::cout << "[IE] exception " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddInput(uint32_t index) {
  const Operand& operand = model_->operands[index];
  SizeVector dims;
  int32_t result = ConvertDims(operand.dimensions, dims);
  if (result != error_t::NOT_ERROR) {
    return result;
  }
  try {
    auto input_node =
        std::make_shared<op::Parameter>(element::f32, Shape(dims));
    index_op_map_[index] = input_node->output(0);
    ngraph_inputs_.push_back(input_node);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add input layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddOutput(uint32_t index) {
  if (index_op_map_.find(index) == index_op_map_.end()) {
    std::cout << "Layer for output index " << index << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto output_node = std::make_shared<op::Result>(index_op_map_[index]);
    ngraph_outputs_.push_back(output_node);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add output layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConstant(uint32_t index,
                                 std::vector<size_t> group_conv_weights_dims) {
  try {
    const Operand& operand = model_->operands[index];
    SizeVector dims;
    int32_t result = ConvertDims(operand.dimensions, dims);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
    size_t rank = dims.size();
    Layout layout =
        rank == 1
            ? Layout::C
            : rank == 2 ? Layout::HW : rank == 3 ? Layout::CHW : Layout::NCHW;
    // GNA also only support FP32 representing with PREFER_ULTRA_LOW_POWER.
    bool fp32_precision = preference_ != PREFER_LOW_POWER;
    Blob::Ptr blob;
    if (fp32_precision) {
      // GNA only accepts FP32 precision, cpu/gpu use FP32 currently.
      blob = make_shared_blob<float>({Precision::FP32, dims, layout});
    } else {
      // MYRIAD only accepts FP16 precision.
      blob = make_shared_blob<int16_t>({Precision::FP16, dims, layout});
    }
    blob->allocate();
    const float* src =
        reinterpret_cast<const float*>(model_->values[index].buffer);
    std::shared_ptr<op::Constant> constant_node;
    SizeVector const_dims =
        group_conv_weights_dims.size() == 0 ? dims : group_conv_weights_dims;
    if (fp32_precision) {
      float* dst = blob->buffer().as<float*>();
      result = Reorder<float>(dst, src, operand.dimensions);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      constant_node = std::make_shared<op::Constant>(
          element::f32, Shape(const_dims), blob->buffer().as<float*>());
    } else {
      int16_t* dst = blob->buffer().as<int16_t*>();
      result = Reorder<int16_t>(dst, src, operand.dimensions);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
      constant_node = std::make_shared<op::Constant>(
          element::i16, Shape(const_dims), blob->buffer().as<int16_t*>());
    }
    index_op_map_[index] = constant_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add constant " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

// int32_t Compilation::AddActivationByFusedCode(int32_t fuse_code,
//                                               size_t input_layer,
//                                               const std::string& name,
//                                               size_t& activiation_layer_id) {
//   try {
//     if (fuse_code == FUSED_RELU) {
//       activiation_layer_id =
//           builder_->addLayer({{input_layer}}, Builder::ReLULayer(name));
//     } else if (fuse_code == FUSED_RELU1) {
//       activiation_layer_id = builder_->addLayer(
//           {{input_layer}},
//           Builder::ClampLayer(name).setMinValue(-1.0).setMaxValue(1.0));
//     } else if (fuse_code == FUSED_RELU6) {
//       activiation_layer_id = builder_->addLayer(
//           {{input_layer}},
//           Builder::ClampLayer(name).setMinValue(0.0).setMaxValue(6.0));
//     } else {
//       std::cout << "Fuse code " << fuse_code << " is not supported";
//       return error_t::BAD_DATA;
//     }
//   } catch (const std::exception& ex) {
//     std::cout << "[IE] failed to add relu layer " << ex.what();
//     return error_t::OP_FAILED;
//   }
//   return error_t::NOT_ERROR;
// }

int32_t Compilation::AddElementwise(const Operation& operation) {
  // Setup element-wise parameters.
  ElementWiseParams params;
  int32_t result = GetElementWiseParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  for (size_t i = 0; i < 2; ++i) {
    const uint32_t input_index = operation.inputs[i];
    // Setup constants
    if (model_->values.find(input_index) != model_->values.end()) {
      result = AddConstant(input_index);
      if (result != error_t::NOT_ERROR) {
        return result;
      }
    }
  }
  try {
    std::shared_ptr<ngraph::Node> elementwise_node;
    auto input0 = index_op_map_[operation.inputs[0]];
    auto input1 = index_op_map_[operation.inputs[1]];
    if (operation.type == ADD) {
      elementwise_node = std::make_shared<op::v1::Add>(input0, input1);
    } else if (operation.type == MUL) {
      elementwise_node = std::make_shared<op::v1::Multiply>(input0, input1);
    } else {
      std::cout << "Operation type " << operation.type << " is not supported";
      return error_t::BAD_DATA;
    }
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = elementwise_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add elementwise layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddConvolution(const Operation& operation) {
  // Setup convolution params.
  ConvParams params;
  int32_t result = GetConvParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    std::vector<ptrdiff_t> padBegin{params.padding_top, params.padding_left};
    std::vector<ptrdiff_t> padEnd{params.padding_bottom, params.padding_right};
    std::vector<size_t> strides{params.stride_height, params.stride_width};
    std::vector<size_t> dilations{params.dilation_height,
                                  params.dilation_width};
    const uint32_t weights_index = operation.inputs[1];
    const uint32_t bias_index = operation.inputs[2];
    std::shared_ptr<ngraph::Node> conv_node;

    if (params.depthwise) {
      if (params.depthwise_multiplier != 1) {
        std::cout << "depthwise_multiplier " << params.depthwise_multiplier
                  << " is not supported";
        return error_t::BAD_DATA;
      }

      if (params.output_channel !=
          params.input_channel * params.depthwise_multiplier) {
        std::cout
            << "Failed assertion: outputFeatureChannels "
            << params.output_channel
            << " in IE depthwise convolution descriptor must be multiplie of "
               "inFeatureChannels "
            << params.input_channel;
        return error_t::BAD_DATA;
      }
      SizeVector weights_dims{params.depth_out, 1, 1, params.filter_height,
                              params.filter_width};
      AddConstant(weights_index, weights_dims);
      conv_node = std::make_shared<op::v1::GroupConvolution>(
          index_op_map_[input_index], index_op_map_[weights_index],
          Strides(strides), CoordinateDiff(padBegin), CoordinateDiff(padEnd),
          Strides(dilations));
    } else {
      AddConstant(weights_index);
      conv_node = std::make_shared<op::v1::Convolution>(
          index_op_map_[input_index], index_op_map_[weights_index],
          Strides(strides), CoordinateDiff(padBegin), CoordinateDiff(padEnd),
          Strides(dilations));
    }
    // add bias and fused-code
    const uint32_t output_index = operation.outputs[0];
    AddBiasWithBroadcasting(bias_index, output_index);
    auto add_node = std::make_shared<op::v1::Add>(conv_node->output(0),
                                                  index_op_map_[bias_index]);
    index_op_map_[output_index] = add_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add Convolution layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

// int32_t Compilation::AddPooling(const Operation& operation) {
//   // Setup pooling params.
//   PoolingParams params;
//   int32_t result = GetPoolingParams(model_, operation, params);
//   if (result != error_t::NOT_ERROR)
//     return result;
//   const uint32_t input_index = operation.inputs[0];
//   if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
//     std::cout << "The layer for operand index " << input_index
//               << " is not ready";
//     return error_t::BAD_DATA;
//   }
//   try {
//     const uint32_t output_index = operation.outputs[0];
//     std::string output_name(std::to_string(output_index));
//     std::string name(output_name);
//     if (params.fuse_code != FUSED_NONE) {
//       name = name + "_pre_fuse";
//     }
//     const size_t input_layer_id = layer_id_map_[input_index];
//     size_t layer_id = builder_->addLayer(
//         {{input_layer_id}},
//         Builder::PoolingLayer(name)
//             .setExcludePad(true)
//             .setKernel({params.filter_height, params.filter_width})
//             .setStrides({params.stride_width, params.stride_height})
//             .setPaddingsBegin({params.padding_top, params.padding_left})
//             .setPaddingsEnd({params.padding_bottom, params.padding_right})
//             .setPoolingType(operation.type == AVERAGE_POOL_2D
//                                 ? Builder::PoolingLayer::PoolingType::AVG
//                                 : Builder::PoolingLayer::PoolingType::MAX)
//             .setRoundingType(Builder::PoolingLayer::RoundingType::FLOOR));
//     if (params.fuse_code != FUSED_NONE) {
//       result = AddActivationByFusedCode(params.fuse_code, layer_id,
//       output_name,
//                                         layer_id);
//       if (result != error_t::NOT_ERROR) {
//         return result;
//       }
//     }
//     layer_id_map_[output_index] = layer_id;
//   } catch (const std::exception& ex) {
//     std::cout << "[IE] failed to add pooling layer " << ex.what();
//     return error_t::OP_FAILED;
//   }
//   return error_t::NOT_ERROR;
// }

// int32_t Compilation::AddSoftmax(const Operation& operation) {
//   // Setup softmax params.
//   SoftmaxParams params;
//   int32_t result = GetSoftmaxParams(model_, operation, params);
//   if (result != error_t::NOT_ERROR)
//     return result;
//   // Check beta.
//   if (params.beta != 1.0) {
//     std::cout << "beta " << params.beta << " is not supported.";
//     return error_t::BAD_DATA;
//   }
//   const uint32_t input_index = operation.inputs[0];
//   if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
//     std::cout << "The layer for operand index " << input_index
//               << " is not ready";
//     return error_t::BAD_DATA;
//   }
//   try {
//     const uint32_t output_index = operation.outputs[0];
//     std::string name(std::to_string(output_index));
//     const size_t input_layer_id = layer_id_map_[input_index];
//     size_t layer_id = builder_->addLayer(
//         {{input_layer_id}}, Builder::SoftMaxLayer(name).setAxis(1));
//     layer_id_map_[output_index] = layer_id;
//   } catch (const std::exception& ex) {
//     std::cout << "[IE] failed to add softmax layer " << ex.what();
//     return error_t::OP_FAILED;
//   }
//   return error_t::NOT_ERROR;
// }

int32_t Compilation::AddReshape(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation.outputs[0];
    const Operand& output = model_->operands[output_index];
    SizeVector dims;
    int32_t result = ConvertDims(output.dimensions, dims);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
    auto target_shape_node = std::make_shared<op::Constant>(
        element::i64, Shape{output.dimensions.size()}, dims);
    auto reshape_node = std::make_shared<op::v1::Reshape>(
        index_op_map_[input_index], target_shape_node->output(0), true);
    index_op_map_[output_index] = reshape_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add reshape layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

// int32_t Compilation::AddConcatenation(const Operation& operation) {
//   // Setup concatenation params.
//   ConcatParams params;
//   int32_t result = GetConcatParams(model_, operation, params);
//   if (result != error_t::NOT_ERROR)
//     return result;

//   std::vector<PortInfo> input_port_infos;
//   std::vector<Port> input_ports;
//   for (size_t i = 0; i < operation.inputs.size() - 1; ++i) {
//     const uint32_t input_index = operation.inputs[i];
//     if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
//       // Setup constants
//       if (model_->values.find(input_index) != model_->values.end()) {
//         result = AddConstant(input_index);
//         if (result != error_t::NOT_ERROR) {
//           return result;
//         }
//       } else {
//         std::cout << "The layer for operand index " << input_index
//                   << " is not ready";
//         return error_t::BAD_DATA;
//       }
//     }
//     const size_t layer_id = layer_id_map_[input_index];
//     input_port_infos.push_back({layer_id});
//     input_ports.push_back(builder_->getLayer(layer_id)->getOutputPorts()[0]);
//   }

//   int axis = 0;
//   const uint32_t rank =
//   model_->operands[operation.inputs[0]].dimensions.size(); if (rank == 1 ||
//   rank == 2) {
//     axis = params.axis;
//   } else if (rank == 3) {
//     // HWC -> CHW
//     if (params.axis == 0) {
//       axis = 1;
//     } else if (params.axis == 1) {
//       axis = 2;
//     } else if (params.axis == 2) {
//       axis = 0;
//     }
//   } else if (rank == 4) {
//     // NHWC -> NCHW
//     if (params.axis == 0) {
//       axis = 0;
//     } else if (params.axis == 1) {
//       axis = 2;
//     } else if (params.axis == 2) {
//       axis = 3;
//     } else if (params.axis == 3) {
//       axis = 1;
//     }
//   } else {
//     std::cout << "rank " << rank << " is not supported.";
//     return error_t::BAD_DATA;
//   }

//   const uint32_t output_index = operation.outputs[0];
//   std::string name(std::to_string(output_index));
//   try {
//     size_t layer_id = builder_->addLayer(
//         input_port_infos,
//         Builder::ConcatLayer(name).setInputPorts(input_ports).setAxis(axis));
//     layer_id_map_[output_index] = layer_id;
//   } catch (const std::exception& ex) {
//     std::cout << "[IE] failed to add concat layer " << ex.what();
//     return error_t::OP_FAILED;
//   }
//   return error_t::NOT_ERROR;
// }

int32_t Compilation::AddFullyConnected(const Operation& operation) {
  // Setup fully connected params.
  FullyConnectedParams params;
  int32_t result = GetFullyConnectedParams(model_, operation, params);
  if (result != error_t::NOT_ERROR)
    return result;
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    const uint32_t weights_index = operation.inputs[1];
    const uint32_t bias_index = operation.inputs[2];
    AddConstant(weights_index);
    AddConstant(bias_index);
    std::vector<int32_t> dims{params.input_batch_size, params.input_size};
    auto target_shape_node =
        std::make_shared<op::Constant>(element::i64, Shape{2}, dims);
    auto reshaped_input_node = std::make_shared<op::v1::Reshape>(
        index_op_map_[input_index], target_shape_node->output(0), true);
    // Need to transpose the weights.
    auto matmul_node = std::make_shared<op::v0::MatMul>(
        reshaped_input_node->output(0), index_op_map_[weights_index], false,
        true);
    auto add_node = std::make_shared<op::v1::Add>(matmul_node->output(0),
                                                  index_op_map_[bias_index]);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = add_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add fc layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

// int32_t Compilation::AddResizeBilinear(const Operation& operation) {
//   // Setup resize bilinear params.
//   ResizeBilinearParams params;
//   int32_t result = GetResizeBilinearParams(model_, operation, params);
//   if (result != error_t::NOT_ERROR)
//     return result;

//   const uint32_t input_index = operation.inputs[0];
//   if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
//     std::cout << "The layer for operand index " << input_index
//               << " is not ready";
//     return error_t::BAD_DATA;
//   }

//   try {
//     const uint32_t output_index = operation.outputs[0];
//     std::string name(std::to_string(output_index));
//     const size_t input_layer_id = layer_id_map_[input_index];
//     size_t layer_id = builder_->addLayer(
//         {{input_layer_id}},
//         Builder::ResampleLayer(name)
//             .setResampleType("caffe.ResampleParameter.LINEAR")
//             .setAntialias(false)
//             .setFactor(params.x_scale)
//             .setWidth(params.width)
//             .setHeight(params.height));
//     layer_id_map_[output_index] = layer_id;
//   } catch (const std::exception& ex) {
//     std::cout << "[IE] failed to add resize bilinear layer " << ex.what();
//     return error_t::OP_FAILED;
//   }
//   return error_t::NOT_ERROR;
// }

// int32_t Compilation::AddArgmax(const Operation& operation) {
//   ArgmaxParams params;
//   int32_t result = GetArgmaxParams(model_, operation, params);
//   if (result != error_t::NOT_ERROR)
//     return error_t::BAD_DATA;

//   if (params.axis != 3) {
//     std::cout << "Only support axis for channel.";
//     return error_t::BAD_DATA;
//   }
//   const uint32_t input_index = operation.inputs[0];
//   if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
//     std::cout << "The layer for operand index " << input_index
//               << " is not ready";
//     return error_t::BAD_DATA;
//   }
//   try {
//     const uint32_t output_index = operation.outputs[0];
//     std::string name(std::to_string(output_index));
//     const size_t input_layer_id = layer_id_map_[input_index];
//     size_t layer_id = builder_->addLayer(
//         {{input_layer_id}},
//         Builder::ArgMaxLayer(name).setAxis(1).setOutMaxVal(0).setTopK(1));
//     layer_id_map_[output_index] = layer_id;
//   } catch (const std::exception& ex) {
//     std::cout << "[IE] failed to add argmax layer " << ex.what();
//     return error_t::OP_FAILED;
//   }
//   return error_t::NOT_ERROR;
// }

int32_t Compilation::AddSigmoid(const Operation& operation) {
  const uint32_t input_index = operation.inputs[0];
  if (index_op_map_.find(input_index) == index_op_map_.end()) {
    std::cout << "The layer for operand index " << input_index
              << " is not ready";
    return error_t::BAD_DATA;
  }
  try {
    auto sigmoid_node =
        std::make_shared<op::v0::Sigmoid>(index_op_map_[input_index]);
    const uint32_t output_index = operation.outputs[0];
    index_op_map_[output_index] = sigmoid_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add sigmoid layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

int32_t Compilation::AddBiasWithBroadcasting(uint32_t bias_index,
                                             uint32_t output_index) {
  try {
    AddConstant(bias_index);
    const Operand& output = model_->operands[output_index];
    SizeVector dims;
    uint32_t result = ConvertDims(output.dimensions, dims);
    if (result != error_t::NOT_ERROR) {
      return result;
    }
    // Need to broadcast the bias for convolution paticularly
    auto target_shape_node =
        std::make_shared<op::Constant>(element::i64, Shape{dims.size()}, dims);
    // Keep the bias's channel size, and broadcast along N,H,W
    auto axis_mapping_node =
        op::Constant::create<int64_t>(element::i64, Shape{1}, {1});
    auto broadcast_node = std::make_shared<op::v1::Broadcast>(
        index_op_map_[bias_index], target_shape_node->output(0),
        axis_mapping_node->output(0));
    index_op_map_[bias_index] = broadcast_node->output(0);
  } catch (const std::exception& ex) {
    std::cout << "[IE] failed to add broadcast layer " << ex.what();
    return error_t::OP_FAILED;
  }
  return error_t::NOT_ERROR;
}

}  // namespace InferenceEngine
