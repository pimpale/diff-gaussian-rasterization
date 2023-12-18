/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
  auto lambda = [&t](size_t N)
  {
    t.resize_({(long long)N});
    return reinterpret_cast<char *>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &colors,
    const torch::Tensor &opacity,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const bool prefiltered,
    const bool debug)
{
  if (means3D.ndimension() != 3 || means3D.size(2) != 3)
  {
    AT_ERROR("means3D must have dimensions (batch, num_points, 3)");
  }


  auto batch = means3D.size(0);
  auto num_points = means3D.size(1);

  if (batch == 0) {
    AT_ERROR("batch size must be > 0");
  }

  if (num_points == 0) {
    AT_ERROR("num_points must be > 0");
  }

  if (background.ndimension() != 1 || background.size(0) != NUM_CHANNELS)
  {
    AT_ERROR("background must have dimensions (3)");
  }

  if (colors.ndimension() != 3 || colors.size(0) != batch || colors.size(1) != num_points || colors.size(2) != NUM_CHANNELS)
  {
    AT_ERROR("colors must have dimensions (batch, num_points, 3)");
  }

  if (opacity.ndimension() != 3 || opacity.size(0) != batch || opacity.size(1) != num_points || opacity.size(2) != 1)
  {
    AT_ERROR("opacity must have dimensions (batch, num_points, 1)");
  }

  if (scales.ndimension() != 3 || scales.size(0) != batch || scales.size(1) != num_points || scales.size(2) != 3)
  {
    AT_ERROR("scales must have dimensions (batch, num_points, 3)");
  }

  if (rotations.ndimension() != 3 || rotations.size(0) != batch || rotations.size(1) != num_points || rotations.size(2) != 4)
  {
    AT_ERROR("rotations must have dimensions (batch, num_points, 4)");
  }

  if (cov3D_precomp.ndimension() != 3 || cov3D_precomp.size(0) != batch || cov3D_precomp.size(1) != num_points || cov3D_precomp.size(2) != 6)
  {
    AT_ERROR("cov3D_precomp must have dimensions (batch, num_points, 6)");
  }

  if (viewmatrix.ndimension() != 2 || viewmatrix.size(0) != 4 || viewmatrix.size(1) != 4)
  {
    AT_ERROR("viewmatrix must have dimensions (4, 4)");
  }

  if (projmatrix.ndimension() != 2 || projmatrix.size(0) != 4 || projmatrix.size(1) != 4)
  {
    AT_ERROR("projmatrix must have dimensions (4, 4)");
  }

  if (campos.ndimension() != 1 || campos.size(0) != 3)
  {
    AT_ERROR("campos must have dimensions (3)");
  }

  if (sh.ndimension() != 4 || sh.size(0) != batch || sh.size(1) != num_points || sh.size(3) != 3)
  {
    AT_ERROR("sh must have dimensions (batch, num_points, M, 3)");
  }

  // m is max_coeffs for SH
  int M = 0;
  if (sh.size(0) != 0)
  {
    M = sh.size(2);
  }

  const int P = num_points;
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({batch, NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({batch, P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

  int rendered = 0;
  if (P != 0)
  {
    for (int b = 0; b < batch; b++)
    {
      std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
      std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
      std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

      rendered += CudaRasterizer::Rasterizer::forward(
          geomFunc,
          binningFunc,
          imgFunc,
          P, degree, M,
          background.contiguous().data<float>(),
          W, H,
          means3D[b].contiguous().data<float>(),
          sh[b].contiguous().data_ptr<float>(),
          colors[b].contiguous().data<float>(),
          opacity[b].contiguous().data<float>(),
          scales[b].contiguous().data_ptr<float>(),
          scale_modifier,
          rotations[b].contiguous().data_ptr<float>(),
          cov3D_precomp[b].contiguous().data<float>(),
          viewmatrix.contiguous().data<float>(),
          projmatrix.contiguous().data<float>(),
          campos.contiguous().data<float>(),
          tan_fovx,
          tan_fovy,
          prefiltered,
          out_color[b].contiguous().data<float>(),
          radii[b].contiguous().data<int>(),
          debug);
    }
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &radii,
    const torch::Tensor &colors,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &dL_dout_color,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &geomBuffer,
    const int R,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0)
  {
    M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if (P != 0)
  {
    CudaRasterizer::Rasterizer::backward(P, degree, M, R,
                                         background.contiguous().data<float>(),
                                         W, H,
                                         means3D.contiguous().data<float>(),
                                         sh.contiguous().data<float>(),
                                         colors.contiguous().data<float>(),
                                         scales.data_ptr<float>(),
                                         scale_modifier,
                                         rotations.data_ptr<float>(),
                                         cov3D_precomp.contiguous().data<float>(),
                                         viewmatrix.contiguous().data<float>(),
                                         projmatrix.contiguous().data<float>(),
                                         campos.contiguous().data<float>(),
                                         tan_fovx,
                                         tan_fovy,
                                         radii.contiguous().data<int>(),
                                         reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
                                         reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
                                         reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
                                         dL_dout_color.contiguous().data<float>(),
                                         dL_dmeans2D.contiguous().data<float>(),
                                         dL_dconic.contiguous().data<float>(),
                                         dL_dopacity.contiguous().data<float>(),
                                         dL_dcolors.contiguous().data<float>(),
                                         dL_dmeans3D.contiguous().data<float>(),
                                         dL_dcov3D.contiguous().data<float>(),
                                         dL_dsh.contiguous().data<float>(),
                                         dL_dscales.contiguous().data<float>(),
                                         dL_drotations.contiguous().data<float>(),
                                         debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor &means3D,
    torch::Tensor &viewmatrix,
    torch::Tensor &projmatrix)
{
  const int P = means3D.size(0);

  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0)
  {
    CudaRasterizer::Rasterizer::markVisible(P,
                                            means3D.contiguous().data<float>(),
                                            viewmatrix.contiguous().data<float>(),
                                            projmatrix.contiguous().data<float>(),
                                            present.contiguous().data<bool>());
  }

  return present;
}