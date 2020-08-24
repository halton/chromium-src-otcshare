// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include "third_party/blink/renderer/modules/ml/v2/nn_model.h"

#include <utility>

#include "mojo/public/cpp/bindings/interface_ptr.h"
#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/neural_network_context.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_compilation.h"
#include "third_party/blink/renderer/modules/ml/v2/nn_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

namespace {

void CopyDataToSharedBuffer(
    const HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    ml::mojom::blink::ModelInfoPtr& model_info) {
  uint32_t total_byte_length = 0;
  for (HeapHashMap<WTF::String, Member<DOMArrayBufferView>>::const_iterator
           itr = buffer_views.begin();
       itr != buffer_views.end(); ++itr) {
    total_byte_length += itr->value->deprecatedByteLengthAsUnsigned();
  }

  mojo::ScopedSharedBufferHandle memory;
  mojo::ScopedSharedBufferMapping mapping;
  if (total_byte_length != 0) {
    memory = mojo::SharedBufferHandle::Create(total_byte_length);
    mapping = memory->Map(total_byte_length);
  }

  uint32_t offset = 0;
  for (WTF::HashMap<WTF::String,
                    ml::mojom::blink::OperandValueInfoPtr>::const_iterator itr =
           model_info->values.begin();
       itr != model_info->values.end(); ++itr) {
    const ml::mojom::blink::OperandValueInfoPtr& value_info = itr->value;
    DOMArrayBufferView* view =
        buffer_views.at(WTF::String::Number(value_info->index));
    uint32_t length = view->deprecatedByteLengthAsUnsigned();
    value_info->offset = offset;
    value_info->length = length;
    uint8_t* base = static_cast<uint8_t*>(mapping.get()) + offset;
    memcpy(static_cast<void*>(base), view->BaseAddress(), length);
    offset += length;
  }

  if (total_byte_length != 0) {
    model_info->memory =
        memory->Clone(mojo::SharedBufferHandle::AccessMode::READ_ONLY);
    model_info->memory_size = total_byte_length;
  }
}

// Traversal graph inorder to create model.
void BuildNeuralNetworkModel(
    Operand* root,
    HeapHashMap<WTF::String, Member<DOMArrayBufferView>>& buffer_views,
    ml::mojom::blink::ModelInfoPtr& model_info,
    HashMap<WTF::String, uint32_t>& name_index) {
  // The index is sequence increase to index all operands in model.
  uint32_t index = 0;
  // the stack is used for traversaling model tree.
  HeapVector<Member<Operand>> stack;
  Operand* operand = root;
  while (operand || !stack.IsEmpty()) {
    while (operand) {
      stack.push_back(operand);
      operand = operand->FirstInput();
    }
    // It will be input/constant if there is no FirstInput operand.
    if (!stack.IsEmpty()) {
      // The index_ in Operand will be add when calling NextInput, so keep the
      // line.
      Operand* next_input = stack.back()->NextInput();
      if (next_input && !next_input->Traversal()) {
        operand = next_input;
      } else {
        // Call the AddLayer virtual function to Add the operand with Android NN
        // API.
        stack.back()->AddLayer(model_info, buffer_views, name_index, index);
        // Set the operand as traversalled so that it doesn't push again.
        stack.back()->SetTraversal(true);
        // Pop the node.
        stack.pop_back();
        // Don't use the code [operand = stack.back()] so that Origin branch
        // [while (operand)] will be not traversalled again.
        // operand = stack.back();
      }
    }
  }

  CopyDataToSharedBuffer(buffer_views, model_info);
}

}  // namespace

NNModel::NNModel(ml::mojom::blink::ModelPtrInfo info,
                 NamedOperandVector* outputs) {
  model_.Bind(std::move(info));
  model_.set_connection_error_handler(
      WTF::Bind(&NNModel::OnConnectionError, WrapWeakPersistent(this)));
  FinishCreatingModel(std::move(outputs));
}

NNModel::~NNModel() = default;

void NNModel::FinishCreatingModel(NamedOperandVector* outputs) {
  ml::mojom::blink::ModelInfoPtr model_info =
      ml::mojom::blink::ModelInfo::New();
  for (uint32_t i = 0; i < outputs->size(); ++i) {
    NamedOperand* output = (*outputs)[i];
    BuildNeuralNetworkModel(output->operand(), buffer_views_, model_info,
                            name_index_);
    // identifyInputsAndOutputs for outputs
    model_info->outputs.push_back(output->operand()->Index());
    // map the name and index that will be used in execution.
    name_index_.insert(output->name(), i);
  }
  model_->Finish(std::move(model_info),
                 WTF::Bind(&NNModel::OnResultCode, WrapPersistent(this)));
  return;
}

void NNModel::OnResultCode(int32_t result_code) {
  LOG(INFO) << "The result code is " << result_code
            << "after finish creating model.";
}

ScriptPromise NNModel::createCompilation(ScriptState* script_state,
                                         const CompilationOptions* options) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!model_) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "Model service unavailable."));
    return promise;
  }

  int32_t preference;
  const WebString& operand_type =
      options->hasPowerPreference() ? options->powerPreference() : "default";
  if (operand_type == "default") {
    preference = NeuralNetworkContext::kPreferFastSingleAnswer;
  } else if (operand_type == "high-performance") {
    preference = NeuralNetworkContext::kPreferSustainedSpeed;
  } else if (operand_type == "low-power") {
    preference = NeuralNetworkContext::kPreferLowPower;
  } else {
    resolver->Reject(
        MakeGarbageCollected<DOMException>(DOMExceptionCode::kNotSupportedError,
                                           "The preference isn't supported."));
    return promise;
  }

  requests_.insert(resolver);

  model_->CreateCompilation(WTF::Bind(&NNModel::OnCreateCompilation,
                                      WrapPersistent(this),
                                      WrapPersistent(resolver), preference));
  return promise;
}

void NNModel::OnCreateCompilation(
    ScriptPromiseResolver* resolver,
    int32_t preference,
    int32_t result_code,
    ml::mojom::blink::CompilationInitParamsPtr init_params) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code == ml::mojom::blink::NOT_ERROR) {
    resolver->Resolve(MakeGarbageCollected<NNCompilation>(
        std::move(init_params->compilation), preference,
        std::move(name_index_)));
  } else {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        "createCompilation fails: " + String::Number(result_code)));
  }
}

void NNModel::Trace(blink::Visitor* visitor) const {
  visitor->Trace(requests_);
  visitor->Trace(buffer_views_);
  ScriptWrappable::Trace(visitor);
}

void NNModel::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "Model is not implemented."));
  }
  requests_.clear();
  model_.reset();
}

}  // namespace blink
