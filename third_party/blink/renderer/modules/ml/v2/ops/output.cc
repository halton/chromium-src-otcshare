// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/ops/output.h"

#include <memory>

namespace blink {

Output::Output(HeapVector<Member<Operand>> inputs)
    : inputs_(std::move(inputs)) {}

Operand* Output::FirstInput() const {
  if (inputs_.IsEmpty())
    return nullptr;
  return inputs_[0];
}
Operand* Output::NextInput() {
  input_index_++;
  if (input_index_ >= inputs_.size())
    return nullptr;
  return inputs_[input_index_];
}

HeapVector<Member<Operand>>& Output::Inputs() {
  return inputs_;
}

void Output::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  Operand::Trace(visitor);
}

}  // namespace blink
