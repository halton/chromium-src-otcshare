// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_EXECUTION_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_EXECUTION_H_

#include <map>
#include <memory>
#include <utility>

#include "mojo/public/cpp/system/buffer.h"
#include "services/ml/public/mojom/compilation.mojom-blink.h"
#include "services/ml/public/mojom/execution.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

struct OutputBufferInfo {
  OutputBufferInfo(uint32_t offset, uint32_t length)
      : offset(offset), length(length) {}
  uint32_t offset;
  uint32_t length;
};

class NNExecution final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NNExecution(ml::mojom::blink::ExecutionInitParamsPtr,
              HashMap<WTF::String, uint32_t>);
  ~NNExecution() override;

  void setInput(const String&,
                MaybeShared<DOMArrayBufferView>,
                ExceptionState&);
  void setOutput(const String&,
                 MaybeShared<DOMArrayBufferView>,
                 ExceptionState&);
  ScriptPromise startCompute(ScriptState*);

  void Trace(blink::Visitor*) const override;

 private:
  ml::mojom::blink::UserBufferPtr CreateUserBuffer();
  void OnResultCode(ScriptPromiseResolver*, const String&, int32_t);
  void OnStartCompute(ScriptPromiseResolver*, int32_t);
  void OnConnectionError();

  ml::mojom::blink::ExecutionPtr execution_;
  mojo::ScopedSharedBufferHandle memory_;
  mojo::ScopedSharedBufferMapping mapping_;
  WTF::Vector<std::unique_ptr<OutputBufferInfo>> output_buffer_info_;

  HeapHashSet<Member<ScriptPromiseResolver>> requests_;
  // The hash map is used in execution phase.
  HashMap<WTF::String, uint32_t> name_index_;
  // Create Shared Buffer in blink to support unspecified output dimensions.
  HeapVector<Member<DOMArrayBufferView>> input_buffer_views_;
  HeapVector<Member<DOMArrayBufferView>> output_buffer_views_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_EXECUTION_H_
