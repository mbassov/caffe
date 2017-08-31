// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/31
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_PRARM_HPP_
#define CAFFE_FRCNN_PRARM_HPP_

#include <vector>
#include <string>

namespace caffe{

class FrcnnParam {
public:

  static float rpn_nms_thresh;
  static int rpn_pre_nms_top_n;
  static int rpn_post_nms_top_n;
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at
  // orig image scale)
  static float rpn_min_size;

  static float test_rpn_nms_thresh;
  static int test_rpn_pre_nms_top_n;
  static int test_rpn_post_nms_top_n;
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at
  // orig image scale)
  static float test_rpn_min_size;

  static int feat_stride;
  static std::vector<float> anchors;
  static int n_classes;
  // ========================================
  static void load_param(const std::string default_config_path);
  static void print_param();
};

}

#endif // CAFFE_FRCNN_PRARM_HPP_
