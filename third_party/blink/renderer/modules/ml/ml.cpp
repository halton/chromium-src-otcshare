// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "third_party/blink/renderer/bindings/modules/v8/neural_network_context_or_nn_context.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/modules/ml/navigator_ml.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_context.h"

namespace blink {

ML::ML(NavigatorML& navigator_ml)
    : ExecutionContextLifecycleObserver(navigator_ml.DomWindow()),
      navigator_ml_(&navigator_ml) {}

ML::~ML() = default;

// Get v1 context.
void ML::getNeuralNetworkContext(WebNNContext& result) {
  getNeuralNetworkContext("v1", result);
}

void ML::getNeuralNetworkContext(const AtomicString& type,
                                 WebNNContext& result) {
  if (type == "v1") {
    if (!v1_context_)
      v1_context_ =
          MakeGarbageCollected<NeuralNetworkContext>(navigator_ml_.Get());
    result.SetNeuralNetworkContext(v1_context_.Get());
  } else if (type == "v2") {
    if (!v2_context_)
      v2_context_ = MakeGarbageCollected<NNContext>(navigator_ml_.Get());
    result.SetNNContext(v2_context_.Get());
  } else {
    LOG(ERROR) << "The " << type << " is not supported.";
  }
}

void ML::ContextDestroyed() {
  Dispose();
}

void ML::Trace(blink::Visitor* visitor) const {
  visitor->Trace(navigator_ml_);
  visitor->Trace(v1_context_);
  visitor->Trace(v2_context_);
  ScriptWrappable::Trace(visitor);
  ContextLifecycleObserver::Trace(visitor);
}

void ML::Dispose() {}

}  // namespace blink
