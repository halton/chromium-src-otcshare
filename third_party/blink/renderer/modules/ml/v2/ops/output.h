// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_OUTPUT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_OUTPUT_H_

#include "third_party/blink/renderer/modules/ml/v2/operand.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"

namespace blink {

class Output : public Operand {
 public:
  explicit Output(HeapVector<Member<Operand>>);
  ~Output() override = default;

  // First/NextInput are used for getting inputs when traversaling model tree.
  Operand* FirstInput() const override;
  Operand* NextInput() override;
  HeapVector<Member<Operand>>& Inputs();

  // Interface required by garbage collection.
  void Trace(Visitor*) const override;

 private:
  // the inputs of operand.
  HeapVector<Member<Operand>> inputs_;
  // current traversalled input index.
  uint32_t input_index_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_OUTPUT_H_
