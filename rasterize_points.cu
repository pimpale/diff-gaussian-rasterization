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
#include <vector>
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

std::tuple<
    int,
    torch::Tensor,
    torch::Tensor,
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>>
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
    const int image_channels,
    const int image_height,
    const int image_width,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const bool prefiltered,
    const bool debug)
{
  bool omit_scale = scales.ndimension() == 1 && scales.size(0) == 0;
  bool omit_rot = rotations.ndimension() == 1 && rotations.size(0) == 0;
  bool omit_cov = cov3D_precomp.ndimension() == 1 && cov3D_precomp.size(0) == 0;
  bool omit_sh = sh.ndimension() == 1 && sh.size(0) == 0;
  bool omit_color = colors.ndimension() == 1 && colors.size(0) == 0;

  if (means3D.ndimension() != 3 || means3D.size(2) != 3)
  {
    AT_ERROR("means3D must have dimensions (batch, num_points, 3)");
  }

  auto batch = means3D.size(0);
  auto num_points = means3D.size(1);

  if (batch == 0)
  {
    AT_ERROR("batch size must be > 0");
  }

  if (num_points == 0)
  {
    AT_ERROR("num_points must be > 0");
  }

  if(image_channels > MAX_NUM_CHANNELS) {
    AT_ERROR("image_channels must be <= MAX_NUM_CHANNELS");
  }

  if(image_channels <= 0) {
    AT_ERROR("image_channels must be > 0");
  }

  if(!omit_sh && image_channels != 3) {
    AT_ERROR("image_channels must be 3 when using SH");
  }

  if (background.ndimension() != 1 || background.size(0) != image_channels)
  {
    AT_ERROR("background must have dimensions (image_channels)");
  }

  if (!omit_color)
  {
    if (colors.ndimension() != 3 || colors.size(0) != batch || colors.size(1) != num_points || colors.size(2) != image_channels)
    {
      AT_ERROR("colors must have dimensions (batch, num_points, image_channels)");
    }
  }
  if (opacity.ndimension() != 3 || opacity.size(0) != batch || opacity.size(1) != num_points || opacity.size(2) != 1)
  {
    AT_ERROR("opacity must have dimensions (batch, num_points, 1)");
  }
  if (!omit_scale)
  {
    if (scales.ndimension() != 3 || scales.size(0) != batch || scales.size(1) != num_points || scales.size(2) != 3)
    {
      AT_ERROR("scales must have dimensions (batch, num_points, 3)");
    }
  }
  if (!omit_rot)
  {
    if (rotations.ndimension() != 3 || rotations.size(0) != batch || rotations.size(1) != num_points || rotations.size(2) != 4)
    {
      AT_ERROR("rotations must have dimensions (batch, num_points, 4)");
    }
  }

  if (!omit_cov)
  {
    if (cov3D_precomp.ndimension() != 3 || cov3D_precomp.size(0) != batch || cov3D_precomp.size(1) != num_points || cov3D_precomp.size(2) != 6)
    {
      AT_ERROR("cov3D_precomp must have dimensions (batch, num_points, 6)");
    }
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

  if (!omit_sh)
  {
    if (sh.ndimension() != 4 || sh.size(0) != batch || sh.size(1) != num_points || sh.size(3) != 3)
    {
      AT_ERROR("sh must have dimensions (batch, num_points, M, 3)");
    }
  }

  // m is max_coeffs for SH
  int M = 0;
  if (sh.size(0) != 0)
  {
    M = sh.size(2);
  }

  const int P = num_points;
  const int C = image_channels;
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({batch, C, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({batch, P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  std::vector<torch::Tensor> geomBufferVec;
  std::vector<torch::Tensor> binningBufferVec;
  std::vector<torch::Tensor> imgBufferVec;

  int rendered = 0;
  for (int b = 0; b < batch; b++)
  {
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));

    std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

    rendered += CudaRasterizer::Rasterizer::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        P, degree, M,
        background.contiguous().data_ptr<float>(),
        C, W, H,
        means3D[b].contiguous().data_ptr<float>(),
        omit_sh
            ? sh.contiguous().data_ptr<float>()
            : sh[b].contiguous().data_ptr<float>(),
        omit_color
            ? colors.contiguous().data_ptr<float>()
            : colors[b].contiguous().data_ptr<float>(),
        opacity[b].contiguous().data_ptr<float>(),
        omit_scale
            ? scales.contiguous().data_ptr<float>()
            : scales[b].contiguous().data_ptr<float>(),
        scale_modifier,
        omit_rot
            ? rotations.contiguous().data_ptr<float>()
            : rotations[b].contiguous().data_ptr<float>(),
        omit_cov
            ? cov3D_precomp.contiguous().data_ptr<float>()
            : cov3D_precomp[b].contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_color[b].data_ptr<float>(),
        radii[b].data_ptr<int>(),
        debug);

    geomBufferVec.push_back(geomBuffer);
    binningBufferVec.push_back(binningBuffer);
    imgBufferVec.push_back(imgBuffer);
  }
  return std::make_tuple(rendered, out_color, radii, geomBufferVec, binningBufferVec, imgBufferVec);
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
    const std::vector<torch::Tensor> &geomBuffer,
    const int R,
    const std::vector<torch::Tensor> &binningBuffer,
    const std::vector<torch::Tensor> &imageBuffer,
    const bool debug)
{
  bool omit_scale = scales.ndimension() == 1 && scales.size(0) == 0;
  bool omit_rot = rotations.ndimension() == 1 && rotations.size(0) == 0;
  bool omit_cov = cov3D_precomp.ndimension() == 1 && cov3D_precomp.size(0) == 0;
  bool omit_sh = sh.ndimension() == 1 && sh.size(0) == 0;
  bool omit_color = colors.ndimension() == 1 && colors.size(0) == 0;

  if (means3D.ndimension() != 3 || means3D.size(2) != 3)
  {
    AT_ERROR("means3D must have dimensions (batch, num_points, 3)");
  }


  auto batch = means3D.size(0);
  auto num_points = means3D.size(1);
  auto image_channels = dL_dout_color.size(1);


  if(image_channels > MAX_NUM_CHANNELS) {
    AT_ERROR("image_channels must be <= MAX_NUM_CHANNELS");
  }

  if(image_channels <= 0) {
    AT_ERROR("image_channels must be > 0");
  }

  if(!omit_sh && image_channels != 3) {
    AT_ERROR("image_channels must be 3 when using SH");
  }

  if (batch == 0)
  {
    AT_ERROR("batch size must be > 0");
  }

  if (num_points == 0)
  {
    AT_ERROR("num_points must be > 0");
  }

  if (background.ndimension() != 1 || background.size(0) != image_channels)
  {
    AT_ERROR("background must have dimensions (image_channels)");
  }

  if (!omit_color)
  {
    if (colors.ndimension() != 3 || colors.size(0) != batch || colors.size(1) != num_points || colors.size(2) != image_channels)
    {
      AT_ERROR("colors must have dimensions (batch, num_points, image_channels)");
    }
  }
  if (!omit_scale)
  {
    if (scales.ndimension() != 3 || scales.size(0) != batch || scales.size(1) != num_points || scales.size(2) != 3)
    {
      AT_ERROR("scales must have dimensions (batch, num_points, 3)");
    }
  }
  if (!omit_rot)
  {
    if (rotations.ndimension() != 3 || rotations.size(0) != batch || rotations.size(1) != num_points || rotations.size(2) != 4)
    {
      AT_ERROR("rotations must have dimensions (batch, num_points, 4)");
    }
  }

  if (!omit_cov)
  {
    if (cov3D_precomp.ndimension() != 3 || cov3D_precomp.size(0) != batch || cov3D_precomp.size(1) != num_points || cov3D_precomp.size(2) != 6)
    {
      AT_ERROR("cov3D_precomp must have dimensions (batch, num_points, 6)");
    }
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

  if (!omit_sh)
  {
    if (sh.ndimension() != 4 || sh.size(0) != batch || sh.size(1) != num_points || sh.size(3) != 3)
    {
      AT_ERROR("sh must have dimensions (batch, num_points, M, 3)");
    }
  }

  if (radii.ndimension() != 2 || radii.size(0) != batch || radii.size(1) != num_points)
  {
    AT_ERROR("radii must have dimensions (batch, num_points)");
  }

  if (dL_dout_color.ndimension() != 4 || dL_dout_color.size(0) != batch || dL_dout_color.size(1) != image_channels)
  {
    AT_ERROR("dL_dout_color must have dimensions (batch, image_channels, H, W)");
  }

  if (geomBuffer.size() != batch)
  {
    AT_ERROR("geomBuffer must have dimensions (batch)");
  }

  if (binningBuffer.size() != batch)
  {
    AT_ERROR("binningBuffer must have dimensions (batch)");
  }

  if (imageBuffer.size() != batch)
  {
    AT_ERROR("imageBuffer must have dimensions (batch)");
  }

  // m is max_coeffs for SH
  int M = 0;
  if (sh.size(0) != 0)
  {
    M = sh.size(2);
  }

  const int P = num_points;
  const int C = image_channels;
  const int H = dL_dout_color.size(2);
  const int W = dL_dout_color.size(3);

  torch::Tensor dL_dmeans3D = torch::zeros({batch, P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({batch, P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({batch, P, C}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({batch, P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({batch, P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({batch, P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({batch, P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({batch, P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({batch, P, 4}, means3D.options());

  for (int b = 0; b < batch; b++)
  {
    CudaRasterizer::Rasterizer::backward(P, degree, M, R,
                                         background.contiguous().data_ptr<float>(),
                                         C, W, H,
                                         means3D[b].contiguous().data_ptr<float>(),
                                         omit_sh
                                             ? sh.contiguous().data_ptr<float>()
                                             : sh[b].contiguous().data_ptr<float>(),
                                         omit_color
                                             ? colors.contiguous().data_ptr<float>()
                                             : colors[b].contiguous().data_ptr<float>(),
                                         omit_scale
                                             ? scales.contiguous().data_ptr<float>()
                                             : scales[b].contiguous().data_ptr<float>(),
                                         scale_modifier,
                                         omit_rot
                                             ? rotations.contiguous().data_ptr<float>()
                                             : rotations[b].contiguous().data_ptr<float>(),
                                         omit_cov
                                             ? cov3D_precomp.contiguous().data_ptr<float>()
                                             : cov3D_precomp[b].contiguous().data_ptr<float>(),
                                         viewmatrix.contiguous().data_ptr<float>(),
                                         projmatrix.contiguous().data_ptr<float>(),
                                         campos.contiguous().data_ptr<float>(),
                                         tan_fovx,
                                         tan_fovy,
                                         radii[b].contiguous().data_ptr<int>(),
                                         reinterpret_cast<char *>(geomBuffer[b].contiguous().data_ptr()),
                                         reinterpret_cast<char *>(binningBuffer[b].contiguous().data_ptr()),
                                         reinterpret_cast<char *>(imageBuffer[b].contiguous().data_ptr()),
                                         dL_dout_color[b].contiguous().data_ptr<float>(),
                                         dL_dmeans2D[b].contiguous().data_ptr<float>(),
                                         dL_dconic[b].contiguous().data_ptr<float>(),
                                         dL_dopacity[b].contiguous().data_ptr<float>(),
                                         dL_dcolors[b].contiguous().data_ptr<float>(),
                                         dL_dmeans3D[b].contiguous().data_ptr<float>(),
                                         dL_dcov3D[b].contiguous().data_ptr<float>(),
                                         dL_dsh[b].contiguous().data_ptr<float>(),
                                         dL_dscales[b].contiguous().data_ptr<float>(),
                                         dL_drotations[b].contiguous().data_ptr<float>(),
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
                                            means3D.contiguous().data_ptr<float>(),
                                            viewmatrix.contiguous().data_ptr<float>(),
                                            projmatrix.contiguous().data_ptr<float>(),
                                            present.contiguous().data_ptr<bool>());
  }

  return present;
}