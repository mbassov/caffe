// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------
#include <cub/cub.cuh>
#include <iomanip>

#include "caffe/layers/frcnn_proposal_layer.hpp"
#include "caffe/util/frcnn_utils.hpp"
#include "caffe/util/frcnn_helper.hpp"
#include "caffe/util/frcnn_param.hpp"
#include "caffe/util/frcnn_gpu_nms.hpp"

namespace caffe {

using std::vector;

__global__ void GetIndex(const int n,int *indices){
  CUDA_KERNEL_LOOP(index , n){
    indices[index] = index;
  }
}

template <typename Dtype>
__global__ void BBoxTransformInv(const int nthreads, const Dtype* const bottom_rpn_bbox,
                                 const int height, const int width, const int feat_stride,
                                 const int im_height, const int im_width,
                                 const int* sorted_indices, const float* anchors,
                                 float* const transform_bbox) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    const int score_idx = sorted_indices[index];
    const int i = score_idx % width;  // width
    const int j = (score_idx % (width * height)) / width;  // height
    const int k = score_idx / (width * height); // channel
    float *box = transform_bbox + index * 4;
    box[0] = anchors[k * 4 + 0] + i * feat_stride;
    box[1] = anchors[k * 4 + 1] + j * feat_stride;
    box[2] = anchors[k * 4 + 2] + i * feat_stride;
    box[3] = anchors[k * 4 + 3] + j * feat_stride;
    const Dtype det[4] = { bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i],
                           bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i],
                           bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i],
                           bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i] };
    float src_w = box[2] - box[0] + 1;
    float src_h = box[3] - box[1] + 1;
    float src_ctr_x = box[0] + 0.5 * src_w;
    float src_ctr_y = box[1] + 0.5 * src_h;
    float pred_ctr_x = det[0] * src_w + src_ctr_x;
    float pred_ctr_y = det[1] * src_h + src_ctr_y;
    float pred_w = exp(det[2]) * src_w;
    float pred_h = exp(det[3]) * src_h;
    box[0] = pred_ctr_x - 0.5 * pred_w;
    box[1] = pred_ctr_y - 0.5 * pred_h;
    box[2] = pred_ctr_x + 0.5 * pred_w;
    box[3] = pred_ctr_y + 0.5 * pred_h;
    box[0] = max(0.0f, min(box[0], im_width - 1.0));
    box[1] = max(0.0f, min(box[1], im_height - 1.0));
    box[2] = max(0.0f, min(box[2], im_width - 1.0));
    box[3] = max(0.0f, min(box[3], im_height - 1.0));
  }
}

__global__ void SelectBox(const int nthreads, const float *box, float min_size,
                          int *flags) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    if ((box[index * 4 + 2] - box[index * 4 + 0] < min_size) ||
        (box[index * 4 + 3] - box[index * 4 + 1] < min_size)) {
      flags[index] = 0;
    } else {
      flags[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SelectBoxByIndices(const int nthreads, const float *in_box, int *selected_indices,
                          float *out_box, const Dtype *in_score, Dtype *out_score) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    if ((index == 0 && selected_indices[index] == 1) ||
        (index > 0 && selected_indices[index] == selected_indices[index - 1] + 1)) {
      out_box[(selected_indices[index] - 1) * 4 + 0] = in_box[index * 4 + 0];
      out_box[(selected_indices[index] - 1) * 4 + 1] = in_box[index * 4 + 1];
      out_box[(selected_indices[index] - 1) * 4 + 2] = in_box[index * 4 + 2];
      out_box[(selected_indices[index] - 1) * 4 + 3] = in_box[index * 4 + 3];
      if (in_score!=NULL && out_score!=NULL) {
        out_score[selected_indices[index] - 1] = in_score[index];
      }
    }
  }
}

template <typename Dtype>
__global__ void SelectBoxAftNMS(const int nthreads, const float *in_box, int *keep_indices,
                          Dtype *top_data, const Dtype *in_score, Dtype* top_score) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    top_data[index * 5] = 0;
    int keep_idx = keep_indices[index];
    for (int j = 1; j < 5; ++j) {
      top_data[index * 5 + j] = in_box[keep_idx * 4 + j - 1];
    }
    if (top_score != NULL && in_score != NULL) {
      top_score[index] = in_score[keep_idx];
    }
  }
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  Forward_cpu(bottom, top);
  return ;
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FrcnnProposalLayer);

} // namespace caffe
