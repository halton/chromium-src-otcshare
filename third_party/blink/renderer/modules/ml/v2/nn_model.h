// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_MODEL_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_MODEL_H_

#include "services/ml/public/mojom/model.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/v2/compilation_options.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_context.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

class NNModel final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NNModel(ml::mojom::blink::ModelPtrInfo, NamedOperandVector*);
  ~NNModel() override;

  ScriptPromise createCompilation(ScriptState*,
                                  const CompilationOptions* options);

  void Trace(blink::Visitor*) const override;

 private:
  void FinishCreatingModel(NamedOperandVector* outputs);
  void OnResultCode(int32_t);
  void OnCreateCompilation(ScriptPromiseResolver*,
                           int32_t preference,
                           int32_t,
                           ml::mojom::blink::CompilationInitParamsPtr);
  void OnConnectionError();

  ml::mojom::blink::ModelPtr model_;
  HeapHashSet<Member<ScriptPromiseResolver>> requests_;
  HeapHashMap<WTF::String, Member<DOMArrayBufferView>> buffer_views_;
  // name_index_ is used for getting index for setInput/setOutput in execution
  // phase, the index is sequence start from 0.
  HashMap<WTF::String, uint32_t> name_index_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_MODEL_H_
