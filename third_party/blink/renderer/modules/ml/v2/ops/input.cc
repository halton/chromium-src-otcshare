// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/input.h"

namespace blink {

Input::Input(const String& name, const OperandDescriptor* descriptor)
    : Operand({}),
      name_(name),
      descriptor_(const_cast<OperandDescriptor*>(descriptor)) {}

void Input::AddLayer(
    ml::mojom::blink::ModelInfoPtr& model_info,
    HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    HashMap<WTF::String, uint32_t>& name_index,
    uint32_t& index) {
  // Add operand for input.
  uint32_t input_index = index++;
  Operand::SetIndex(input_index);
  AddOperand(descriptor_, model_info);

  // identifyInputsAndOutputs
  model_info->inputs.push_back(input_index);

  // Map the name and index used in execution that is sequence increase from 0.
  name_index.insert(name_, model_info->inputs.size() - 1);
}

void Input::Trace(Visitor* visitor) const {
  visitor->Trace(descriptor_);
  Operand::Trace(visitor);
}

}  // namespace blink
