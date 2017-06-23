// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/layers/frcnn_anchor_target_layer.hpp"
#include "caffe/layers/frcnn/utils.hpp"
#include "caffe/layers/frcnn/param.hpp"
#include "caffe/layers/frcnn/helper.hpp"

namespace caffe {

namespace Frcnn {

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FrcnnAnchorTargetLayer);

} // namespace frcnn

} // namespace caffe