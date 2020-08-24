// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_COMPILATION_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_COMPILATION_H_

#include "services/ml/public/mojom/compilation.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/modules/ml/v2/compilation_options.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/heap/member.h"

namespace blink {

class NNCompilation final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  NNCompilation(ml::mojom::blink::CompilationPtrInfo,
                int32_t preference,
                HashMap<WTF::String, uint32_t>);
  ~NNCompilation() override;

  ScriptPromise finish(ScriptState*);
  ScriptPromise createExecution(ScriptState*);

  void Trace(blink::Visitor*) const override;

 private:
  void OnResultCode(int32_t);
  void OnCreateExecution(ScriptPromiseResolver*,
                         int32_t,
                         ml::mojom::blink::ExecutionInitParamsPtr);
  void OnConnectionError();

  int32_t preference_;
  ml::mojom::blink::CompilationPtr compilation_;
  HeapHashSet<Member<ScriptPromiseResolver>> requests_;
  // The hash map is used in execution phase.
  HashMap<WTF::String, uint32_t> name_index_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_NN_COMPILATION_H_
