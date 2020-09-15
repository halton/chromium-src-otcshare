// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/v2/nn_execution.h"

#include "mojo/public/cpp/bindings/interface_ptr.h"
#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "services/service_manager/public/cpp/interface_provider.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

namespace {

void InvalidIndex(ExceptionState& state) {
  state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                          "Invalid index.");
  return;
}

}  // namespace

NNExecution::NNExecution(ml::mojom::blink::ExecutionInitParamsPtr init_params,
                         HashMap<WTF::String, uint32_t> name_index)
    : name_index_(std::move(name_index)) {
  execution_.Bind(std::move(init_params->execution));
  execution_.set_connection_error_handler(
      WTF::Bind(&NNExecution::OnConnectionError, WrapWeakPersistent(this)));

  input_buffer_views_.resize(init_params->inputs.size());
  output_buffer_views_.resize(init_params->outputs.size());
}

NNExecution::~NNExecution() = default;

void NNExecution::setInput(const String& name,
                           MaybeShared<DOMArrayBufferView> data,
                           ExceptionState& exception_state) {
  if (name_index_.find(name) == name_index_.end()) {
    return InvalidIndex(exception_state);
  }
  uint32_t index = name_index_.find(name)->value;
  if (index >= input_buffer_views_.size()) {
    return InvalidIndex(exception_state);
  }
  // Hold the buffer view for input to support unspecified output dimensions.
  input_buffer_views_[index] = data.View();
}

void NNExecution::setOutput(const String& name,
                            MaybeShared<DOMArrayBufferView> data,
                            ExceptionState& exception_state) {
  if (name_index_.find(name) == name_index_.end()) {
    return InvalidIndex(exception_state);
  }
  uint32_t index = name_index_.find(name)->value;
  if (index >= output_buffer_views_.size()) {
    return InvalidIndex(exception_state);
  }
  // Hold the buffer view for output to support unspecified output dimensions.
  output_buffer_views_[index] = data.View();
}

ml::mojom::blink::UserBufferPtr NNExecution::CreateUserBuffer() {
  auto user_buffer = ml::mojom::blink::UserBuffer::New();
  uint32_t total_byte_length = 0;
  for (auto& input_buffer_view : input_buffer_views_) {
    total_byte_length += input_buffer_view->deprecatedByteLengthAsUnsigned();
  }
  for (auto& output_buffer_view : output_buffer_views_) {
    total_byte_length += output_buffer_view->deprecatedByteLengthAsUnsigned();
  }
  if (total_byte_length == 0)
    return nullptr;

  if (!memory_.is_valid()) {
    memory_ = mojo::SharedBufferHandle::Create(total_byte_length);
    mapping_ = memory_->Map(total_byte_length);
  }
  uint32_t offset = 0;
  for (auto& input_buffer_view : input_buffer_views_) {
    uint32_t length = input_buffer_view->deprecatedByteLengthAsUnsigned();
    user_buffer->inputs.push_back(
        ml::mojom::blink::BufferInfo::New(offset, length));
    uint8_t* base = static_cast<uint8_t*>(mapping_.get());
    memcpy(static_cast<void*>(base + offset), input_buffer_view->BaseAddress(),
           length);
    offset += length;
  }
  for (auto& output_buffer_view : output_buffer_views_) {
    uint32_t length = output_buffer_view->deprecatedByteLengthAsUnsigned();
    user_buffer->outputs.push_back(
        ml::mojom::blink::BufferInfo::New(offset, length));
    if (output_buffer_info_.size() == 0) {
      output_buffer_info_.push_back(
          std::make_unique<OutputBufferInfo>(offset, length));
    }
    offset += length;
  }

  user_buffer->memory =
      memory_->Clone(mojo::SharedBufferHandle::AccessMode::READ_WRITE);
  user_buffer->memory_size = total_byte_length;

  return user_buffer;
}

ScriptPromise NNExecution::startCompute(ScriptState* script_state) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!execution_) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError,
        "Neural Network service unavailable."));
    return promise;
  }

  requests_.insert(resolver);

  execution_->StartCompute(
      CreateUserBuffer(),
      WTF::Bind(&NNExecution::OnStartCompute, WrapPersistent(this),
                WrapPersistent(resolver)));
  return promise;
}

void NNExecution::OnStartCompute(ScriptPromiseResolver* resolver,
                                 int32_t result_code) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code != ml::mojom::blink::NOT_ERROR) {
    return resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        "startCompute fails " + String::Number(result_code)));
  }

  for (wtf_size_t i = 0; i < output_buffer_info_.size(); ++i) {
    DOMArrayBufferView* view = output_buffer_views_.at(i);
    if (view) {
      std::unique_ptr<OutputBufferInfo>& info = output_buffer_info_.at(i);
      uint8_t* base = static_cast<uint8_t*>(mapping_.get());
      memcpy(view->BaseAddress(), static_cast<const void*>(base + info->offset),
             info->length);
    }
  }
  resolver->Resolve(result_code);
}

void NNExecution::OnResultCode(ScriptPromiseResolver* resolver,
                               const String& operation_name,
                               int32_t result_code) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code != ml::mojom::blink::NOT_ERROR) {
    return resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        operation_name + "fails: " + String::Number(result_code)));
  }

  resolver->Resolve(result_code);
}

void NNExecution::Trace(blink::Visitor* visitor) const {
  visitor->Trace(requests_);
  visitor->Trace(output_buffer_views_);
  visitor->Trace(input_buffer_views_);
  ScriptWrappable::Trace(visitor);
}

void NNExecution::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "Execution is not implemented."));
  }

  requests_.clear();
  execution_.reset();
}

}  // namespace blink
