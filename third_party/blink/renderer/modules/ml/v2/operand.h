// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPERAND_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPERAND_H_

#include "base/logging.h"
#include "services/ml/public/mojom/model.mojom-blink.h"
#include "third_party/blink/public/platform/web_string.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/v2/operand_descriptor.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/wtf/hash_map.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

int32_t StringToOperandType(const WebString& operand_type);

void AddFuseOperand(ml::mojom::blink::ModelInfoPtr& model_info);
void AddUnspecifiedOperand(ml::mojom::blink::ModelInfoPtr& model_info);
void AddOperand(const OperandDescriptor* descriptor,
                ml::mojom::blink::ModelInfoPtr& model_info);
void SetOperandValue(
    uint32_t index,
    DOMArrayBufferView* data,
    HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    ml::mojom::blink::ModelInfoPtr& model_info);
void AddOperation(int32_t type,
                  const Vector<uint32_t>& inputs,
                  const Vector<uint32_t>& outputs,
                  ml::mojom::blink::ModelInfoPtr& model_info);
void IdentifyInputsAndOutputs(const Vector<uint32_t>& inputs,
                              const Vector<uint32_t>& outputs,
                              ml::mojom::blink::ModelInfoPtr& model_info);

class Operand : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  Operand(HeapVector<Member<Operand>>);
  ~Operand() override = default;

  // First/NextInput are used for getting inputs when traversaling model tree.
  Operand* FirstInput() const;
  Operand* NextInput();

  // Add operand into Mojo::ModelInfo, operand value to buffer_views that will
  // be copied to Mojo::ModelInfo->memory which is total memory, and map
  // input/output name to index that will be used in execution phase.
  virtual void AddLayer(
      ml::mojom::blink::ModelInfoPtr& model_info,
      HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
      HashMap<WTF::String, uint32_t>& name_index,
      uint32_t& index);
  void SetIndex(uint32_t index);
  uint32_t Index();
  void SetTraversal(bool traversal);
  bool Traversal();
  HeapVector<Member<Operand>>& Inputs();

  // Interface required by garbage collection.
  void Trace(Visitor*) const override;

 private:
  // index of operand in the model.
  uint32_t index_;
  // the inputs of operand.;
  HeapVector<Member<Operand>> inputs_;
  // The traversalled operand doesn't traversal again.
  bool traversalled_;
  // current traversalled input index.
  uint32_t input_index_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPERAND_H_
