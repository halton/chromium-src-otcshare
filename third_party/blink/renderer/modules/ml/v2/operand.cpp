// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/operand.h"

// #include "third_party/blink/renderer/platform/bindings/enumeration_base.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

namespace blink {

Operand::Operand(HeapVector<Member<Operand>> inputs)
    : inputs_(std::move(inputs)), traversalled_(false) {}

Operand* Operand::FirstInput() const {
  if (inputs_.IsEmpty())
    return nullptr;
  return inputs_[0];
}
Operand* Operand::NextInput() {
  input_index_++;
  if (input_index_ >= inputs_.size())
    return nullptr;
  return inputs_[input_index_];
}

void Operand::AddLayer(
    ml::mojom::blink::ModelInfoPtr& model_info,
    HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    HashMap<WTF::String, uint32_t>& name_index,
    uint32_t& index) {
  LOG(INFO) << "Operand.";
}

void Operand::SetIndex(uint32_t index) {
  index_ = index;
}

uint32_t Operand::Index() {
  return index_;
}

void Operand::SetTraversal(bool traversal) {
  traversalled_ = traversal;
}
bool Operand::Traversal() {
  return traversalled_;
}

HeapVector<Member<Operand>>& Operand::Inputs() {
  return inputs_;
}

void Operand::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  ScriptWrappable::Trace(visitor);
}

// Utils functions
int32_t StringToOperandType(const WebString& operand_type) {
  if (operand_type == "float32") {
    return NeuralNetworkContext::kFloat32;
  } else if (operand_type == "int32") {
    return NeuralNetworkContext::kInt32;
  } else if (operand_type == "uint32") {
    return NeuralNetworkContext::kUint32;
  } else if (operand_type == "tensor-float32") {
    return NeuralNetworkContext::kTensorFloat32;
  } else if (operand_type == "tensor-int32") {
    return NeuralNetworkContext::kTensorInt32;
  } else if (operand_type == "tensor-quant8-asymm") {
    return NeuralNetworkContext::kTensorQuant8Asymm;
  } else {
    // float16 and tensor-float16 are not supported in Android NN API design.
    NOTREACHED();
  }
  return 100;
}

void AddFuseOperand(ml::mojom::blink::ModelInfoPtr& model_info) {
  model_info->operands.push_back(ml::mojom::blink::Operand::New(
      static_cast<int32_t>(NeuralNetworkContext::kInt32),
      WTF::Vector<uint32_t>(), 0, 0));
}

void AddUnspecifiedOperand(ml::mojom::blink::ModelInfoPtr& model_info) {
  model_info->operands.push_back(ml::mojom::blink::Operand::New(
      static_cast<int32_t>(NeuralNetworkContext::kTensorFloat32),
      WTF::Vector<uint32_t>(), 0, 0));
}

void AddOperand(const OperandDescriptor* descriptor,
                ml::mojom::blink::ModelInfoPtr& model_info) {
  model_info->operands.push_back(ml::mojom::blink::Operand::New(
      StringToOperandType(descriptor->type()),
      descriptor->hasDimensions() ? descriptor->dimensions()
                                  : WTF::Vector<uint32_t>(),
      descriptor->hasScale() ? descriptor->scale() : 0,
      descriptor->hasZeroPoint() ? descriptor->zeroPoint() : 0));
}

void SetOperandValue(
    uint32_t index,
    DOMArrayBufferView* data,
    HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    ml::mojom::blink::ModelInfoPtr& model_info) {
  WTF::String index_str = WTF::String::Number(index);
  model_info->values.insert(
      index_str, ml::mojom::blink::OperandValueInfo::New(index, 0, 0));
  buffer_views.insert(index_str, data);
}

void AddOperation(int32_t type,
                  const Vector<uint32_t>& inputs,
                  const Vector<uint32_t>& outputs,
                  ml::mojom::blink::ModelInfoPtr& model_info) {
  model_info->operations.push_back(
      ml::mojom::blink::Operation::New(type, inputs, outputs));
}

void IdentifyInputsAndOutputs(const Vector<uint32_t>& inputs,
                              const Vector<uint32_t>& outputs,
                              ml::mojom::blink::ModelInfoPtr& model_info) {
  model_info->inputs = inputs;
  model_info->outputs = outputs;
}

}  // namespace blink
