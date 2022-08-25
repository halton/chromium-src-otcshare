// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/webnn_service_dml_impl.h"

#include "base/no_destructor.h"
#include "content/browser/ml/webnn/dml/mojo_server_dml_impl.h"

namespace content::webnn {

// static
void WebnnServiceDMLImpl::Create(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver) {
  static base::NoDestructor<WebnnServiceDMLImpl> service{std::move(receiver)};
}

WebnnServiceDMLImpl::WebnnServiceDMLImpl(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnServiceDMLImpl::~WebnnServiceDMLImpl() = default;

void WebnnServiceDMLImpl::BindMojoServer(
    mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) {
  MojoServerDMLImpl::Create(std::move(receiver));
}

}  // namespace content::webnn
