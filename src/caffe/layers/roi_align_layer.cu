// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_data[index] = 0;
      continue;
    }

    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    int roi_width = max(static_cast<int>(roi_end_w - roi_start_w + 1), 1);
    int roi_height = max(static_cast<int>(roi_end_h - roi_start_h + 1), 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>((ph) * bin_size_h);
    Dtype wstart = static_cast<Dtype>((pw) * bin_size_w);
    Dtype hend = static_cast<Dtype>((ph + 1) * bin_size_h);
    Dtype wend = static_cast<Dtype>((pw + 1) * bin_size_w);

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0)), 
        static_cast<Dtype>(height));
    hend = min(max(hend + roi_start_h, static_cast<Dtype>(0)), 
        static_cast<Dtype>(height));
    wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0)), 
        static_cast<Dtype>(width));
    wend = min(max(wend + roi_start_w, static_cast<Dtype>(0)), 
        static_cast<Dtype>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    int bottom_index = 0;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype h_stride = (hend - hstart) / 3.0;
    Dtype w_stride = (wend - wstart) / 3.0;
    for (Dtype h = hstart+h_stride; h <= hend-h_stride+0.01; h += max(h_stride, 0.01)) {
      for (Dtype w = wstart+w_stride; w <= wend-w_stride+0.01; w += max(w_stride, 0.01)) {
        bottom_index++;
        int hlow = min(max(static_cast<int>(floor(h)), 0), height-1);
        int hhigh = hlow + 1;
        int wleft = min(max(static_cast<int>(floor(w)), 0), width-1);
        int wright = wleft + 1;
        int topleft = hlow * width + wleft;
        int topright = hlow * width + wright;
        int bottomleft = hhigh * width + wleft;
        int bottomright = hhigh * width + wright;
        
        Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (h - hlow) / (hhigh - hlow);
        Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (w - wleft) / (wright - wleft);
        Dtype value = (1 - alpha) * (1 - beta) * bottom_data[topleft] + alpha * (1 - beta) * bottom_data[bottomleft]
                            + (1 - alpha) * beta * bottom_data[topright] + alpha * beta * bottom_data[bottomright];
        
        if (value > maxval) {
          maxval = value;
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = offset_bottom_rois[1] * spatial_scale;
      int roi_start_h = offset_bottom_rois[2] * spatial_scale;
      int roi_end_w = offset_bottom_rois[3] * spatial_scale;
      int roi_end_h = offset_bottom_rois[4] * spatial_scale;

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w-1.0 && w <= roi_end_w+1.0 &&
                           h >= roi_start_h-1.0 && h <= roi_end_h+1.0);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w + (Dtype)1, (Dtype)1);
      Dtype roi_height = max(roi_end_h - roi_start_h + (Dtype)1, (Dtype)1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          Dtype hstart = static_cast<Dtype>((ph) * bin_size_h);
          Dtype wstart = static_cast<Dtype>((pw) * bin_size_w);
          Dtype hend = static_cast<Dtype>((ph + 1) * bin_size_h);
          Dtype wend = static_cast<Dtype>((pw + 1) * bin_size_w);

          hstart = min(max(hstart + roi_start_h, (Dtype)(0)), (Dtype)(height));
          hend = min(max(hend + roi_start_h, (Dtype)(0)), (Dtype)(height));
          wstart = min(max(wstart + roi_start_w, (Dtype)(0)), (Dtype)(width));
          wend = min(max(wend + roi_start_w, (Dtype)(0)), (Dtype)(width));

          bool in_bin = (w > wstart - 1.0 && w < wend + 1.0 &&
                      h > hstart - 1.0 && h < hend + 1.0);
          if (!in_bin) {
            continue;
          }

          const int pool_index = ph * pooled_width + pw;
          int bottom_index = 0;
          Dtype h_stride = (hend - hstart) / 3.0;
          Dtype w_stride = (wend - wstart) / 3.0;
          for (Dtype rh = hstart+h_stride; rh <= hend-h_stride+0.01; rh += max(h_stride, 0.01)) {
            for (Dtype rw = wstart+w_stride; rw <= wend-w_stride+0.01; rw += max(w_stride, 0.01)) {
              bottom_index ++;
              if (offset_argmax_data[pool_index] != bottom_index) continue;
              // compute the integer coordinates around (h, w) for bilinear interpolation
              int hlow = min(max(static_cast<int>(floor(rh)), 0), height-1);
              int hhigh = hlow + 1;
              int wleft = min(max(static_cast<int>(floor(rw)), 0), width-1);
              int wright = wleft + 1;
              if (h != hlow && h != hhigh && w != wleft && w != wright) // (w, h) is not around (rw, rh)
                  continue;
              
              Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (rh - hlow) / (hhigh - hlow);
              Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (rw - wleft) / (wright - wleft);
              if (h == hlow && w == wleft) gradient += offset_top_diff[pool_index] * (1 - alpha) * (1 - beta);
              else if (h == hlow && w == wright) gradient += offset_top_diff[pool_index] * (1 - alpha) * beta;
              else if (h == hhigh && w == wleft) gradient += offset_top_diff[pool_index] * alpha * (1 - beta);
              else if (h == hhigh && w == wright) gradient += offset_top_diff[pool_index] * alpha * beta;
            }
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
