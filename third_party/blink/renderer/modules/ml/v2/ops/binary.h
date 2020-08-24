// Copyright 2020 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_BINARY_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_BINARY_H_

#include <memory>

#include "services/ml/public/mojom/model.mojom-blink.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/v2/operand.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/wtf/hash_map.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

enum BinaryType {
  kBinaryTypeAdd = 0,
  kBinaryTypeSub,
  kBinaryTypeMul,
  kBinaryTypeDiv,
  kBinaryTypeMax,
  kBinaryTypeMin,
};

class Binary final : public Operand {
 public:
  Binary(BinaryType, Operand*, Operand*);
  ~Binary() override = default;

  void AddLayer(
      ml::mojom::blink::ModelInfoPtr& model_info,
      HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
      HashMap<WTF::String, uint32_t>& name_index,
      uint32_t& index) override;

 private:
  BinaryType type_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_OPS_BINARY_H_
