// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/graph_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/GraphDML.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::BuildResult;
using ml::webnn::mojom::ComputeResult;
using ml::webnn::mojom::Graph;

}  // namespace

// static
void GraphDMLImpl::Create(mojo::PendingReceiver<Graph> receiver) {
  mojo::MakeSelfOwnedReceiver<Graph>(base::WrapUnique(new GraphDMLImpl()),
                                     std::move(receiver));
}

GraphDMLImpl::~GraphDMLImpl() = default;

GraphDMLImpl::GraphDMLImpl()
    : native_graph_dml_(std::make_unique<GraphDMLNativeImpl>()) {}

void GraphDMLImpl::AddInput(const std::string& name,
                            OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddInput(name, std::move(desc));
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc,
                               const std::vector<uint8_t>& array_buffer) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddConstant(std::move(desc), array_buffer);
}

void GraphDMLImpl::AddElementWiseBinary(uint32_t a_id,
                                        uint32_t b_id,
                                        BinaryOperandType type,
                                        OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddElementWiseBinary(a_id, b_id, type,
                                          std::move(output_desc));
}

void GraphDMLImpl::AddClamp(uint32_t input_id,
                            ClampOptionsPtr options,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddClamp(input_id, std::move(options),
                              std::move(output_desc));
}

void GraphDMLImpl::AddConv2d(uint32_t input_id,
                             uint32_t filter_id,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddConv2d(input_id, filter_id, std::move(options),
                               std::move(output_desc));
}

void GraphDMLImpl::AddReshape(uint32_t input_id,
                              OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddReshape(input_id, std::move(output_desc));
}

void GraphDMLImpl::AddGemm(uint32_t a_id,
                           uint32_t b_id,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddGemm(a_id, b_id, std::move(options),
                             std::move(output_desc));
}

void GraphDMLImpl::AddPool2d(uint32_t input_id,
                             Pool2dOptionsPtr options,
                             Pool2dType type,
                             OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddPool2d(input_id, std::move(options), type,
                               std::move(output_desc));
}

void GraphDMLImpl::AddUnary(uint32_t input_id,
                            UnaryOperandType type,
                            OperandDescriptorPtr output_desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddUnary(input_id, type, std::move(output_desc));
}

void GraphDMLImpl::AddFusionClamp(ClampOptionsPtr options,
                                  uint32_t operator_id) {
  native_graph_dml_->AddFusionClamp(std::move(options), operator_id);
}

void GraphDMLImpl::Build(
    const base::flat_map<std::string, uint32_t>& named_operands,
    BuildCallback callback) {
  std::move(callback).Run(native_graph_dml_->CompileImpl(named_operands));
}

void GraphDMLImpl::Compute(NamedInputsPtr named_inputs,
                           ComputeCallback callback) {
  auto named_outputs = ml::webnn::mojom::NamedOutputs::New();
  ComputeResult result =
      native_graph_dml_->ComputeImpl(std::move(named_inputs), named_outputs);
  std::move(callback).Run(result, std::move(named_outputs));
}

}  // namespace content::webnn
