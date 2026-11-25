// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "rocm_ops.hpp"

namespace aiter {
void qr_reduce_scatter(int64_t handle,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& output,
                       bool cast_bf2half);
void qr_reduce_scatter_route(int64_t handle,
                             const aiter_tensor_t& route,
                             const aiter_tensor_t& output,
                             int64_t rows,
                             int64_t topk,
                             int64_t hidden);
} // namespace aiter

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("qr_reduce_scatter",
          &aiter::qr_reduce_scatter,
          py::arg("handle"),
          py::arg("input"),
          py::arg("output"),
          py::arg("cast_bf2half") = true);
    m.def("qr_reduce_scatter_route",
          &aiter::qr_reduce_scatter_route,
          py::arg("handle"),
          py::arg("route"),
          py::arg("output"),
          py::arg("rows"),
          py::arg("topk"),
          py::arg("hidden"));
}
