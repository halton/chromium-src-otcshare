// Copyright 2021 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context.h"

#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/core/frame/navigator.h"
#include "third_party/blink/renderer/modules/ml/ml.h"

namespace blink {

class MLContextTest : public testing::Test {
 public:
  MLContextTest() = default;
  ~MLContextTest() override = default;
};

TEST_F(MLContextTest, createContextSyncTest) {
  V8TestingScope v8_scope;
  auto* ml = ML::ml(*v8_scope.GetWindow().navigator());
  MLContextOptions* context_options = MLContextOptions::Create();
  context_options->setDevicePreference(V8MLDevicePreference::Enum::kAuto);
  context_options->setPowerPreference(V8MLPowerPreference::Enum::kAuto);
  context_options->setModelFormat(V8MLModelFormat::Enum::kTflite);
  context_options->setNumThreads(100);
  MLContext* context =
      ml->createContextSync(context_options, v8_scope.GetExceptionState());
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(v8_scope.GetExceptionState().Code(),
            ToExceptionCode(DOMExceptionCode::kNoError));
}

TEST_F(MLContextTest, createContextAsyncTest) {
  V8TestingScope v8_scope;
  auto* ml = ML::ml(*v8_scope.GetWindow().navigator());
  MLContextOptions* context_options = MLContextOptions::Create();
  context_options->setDevicePreference(V8MLDevicePreference::Enum::kCpu);
  context_options->setPowerPreference(
      V8MLPowerPreference::Enum::kHighPerformance);
  context_options->setModelFormat(V8MLModelFormat::Enum::kTflite);
  context_options->setNumThreads(100);
  const ScriptPromise promise = ml->createContext(
      v8_scope.GetScriptState(), context_options, v8_scope.GetExceptionState());
  EXPECT_EQ(promise.V8Promise()->State(), v8::Promise::kFulfilled);
}
}  // namespace blink