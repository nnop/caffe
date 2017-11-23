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
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_param.pooled_h();
  pooled_width_ = roi_align_param.pooled_w();
  spatial_scale_ = roi_align_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int* argmax_data = max_idx_.mutable_cpu_data();

  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    // Force malformed ROIs to be 1x1
    int roi_width = max(static_cast<int>(roi_end_w - roi_start_w + 1), 1);
    int roi_height = max(static_cast<int>(roi_end_h - roi_start_h + 1), 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height_);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width_);

    const Dtype* batch_data = bottom[0]->cpu_data() + bottom[0]->offset(roi_batch_ind);
    for (int c = 0; c < channels_; ++c) {
      // channel c bottom_data
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          Dtype hstart = static_cast<Dtype>((ph) * bin_size_h);
          Dtype wstart = static_cast<Dtype>((pw) * bin_size_w);
          Dtype hend = static_cast<Dtype>((ph + 1) * bin_size_h);
          Dtype wend = static_cast<Dtype>((pw + 1) * bin_size_w);

          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0)), 
              static_cast<Dtype>(height_));
          hend = min(max(hend + roi_start_h, static_cast<Dtype>(0)), 
              static_cast<Dtype>(height_));
          wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0)), 
              static_cast<Dtype>(width_));
          wend = min(max(wend + roi_start_w, static_cast<Dtype>(0)), 
              static_cast<Dtype>(width_));

          // Define an empty pooling region to be zero
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          Dtype maxval = is_empty ? 0 : -FLT_MAX;
          // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
          int maxidx = -1;
          const int pool_index = ph * pooled_width_ + pw;
          // Sample 2 points inside each bin
          Dtype h_stride = (hend - hstart) / 3.0;
          Dtype w_stride = (wend - wstart) / 3.0;
          int bottom_index = 0;
          for (Dtype h = hstart+h_stride; h <= hend-h_stride+0.01; h+=max<Dtype>(h_stride, 0.01)) {
            for (Dtype w = wstart+w_stride; w <=wend-w_stride+0.01; w+=max<Dtype>(w_stride, 0.01)) {
              bottom_index++;
              int hlow = min(max(static_cast<int>(floor(h)), 0), height_-1);
              int hhigh = hlow + 1;
              int wleft = min(max(static_cast<int>(floor(w)), 0), width_-1);
              int wright = wleft + 1;
              int topleft = hlow * width_ + wleft;
              int topright = hlow * width_ + wright;
              int bottomleft = hhigh * width_ + wleft;
              int bottomright = hhigh * width_ + wright;

              Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (h - hlow) / (hhigh - hlow);
              Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (w - wleft) / (wright - wleft);
              Dtype value = (1 - alpha) * (1 - beta) * batch_data[topleft] + alpha * (1 - beta) * batch_data[bottomleft]
                                  + (1 - alpha) * beta * batch_data[topright] + alpha * beta * bottom_data[bottomright];

              if (value > maxval) {
                maxval = value;
                maxidx = bottom_index;
              }
            } // w
          } // h
          top_data[pool_index] = maxval;
          argmax_data[pool_index] = maxidx;
        } // pw
      } // ph

      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    } // c
    // Increment one roi
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
