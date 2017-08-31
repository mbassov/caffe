#include "caffe/util/frcnn_utils.hpp"
#include "caffe/util/frcnn_param.hpp"
#include "caffe/common.hpp"

namespace caffe {

using namespace caffe;

float FrcnnParam::rpn_nms_thresh;
int FrcnnParam::rpn_pre_nms_top_n;
int FrcnnParam::rpn_post_nms_top_n;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float FrcnnParam::rpn_min_size;

float FrcnnParam::test_rpn_nms_thresh;
int FrcnnParam::test_rpn_pre_nms_top_n;
int FrcnnParam::test_rpn_post_nms_top_n;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float FrcnnParam::test_rpn_min_size;

int FrcnnParam::feat_stride;
std::vector<float> FrcnnParam::anchors;
int FrcnnParam::n_classes;

void FrcnnParam::load_param(const std::string default_config_path) {

  str_map default_map = parse_json_config(default_config_path);

  FrcnnParam::rpn_nms_thresh = extract_float("rpn_nms_thresh", default_map);
  FrcnnParam::rpn_pre_nms_top_n = extract_int("rpn_pre_nms_top_n", default_map);
  FrcnnParam::rpn_post_nms_top_n = extract_int("rpn_post_nms_top_n", default_map);
  FrcnnParam::rpn_min_size = extract_float("rpn_min_size", default_map);

  // ======================================== Test

  FrcnnParam::test_rpn_nms_thresh = extract_float("test_rpn_nms_thresh", default_map);
  FrcnnParam::test_rpn_pre_nms_top_n = extract_int("test_rpn_pre_nms_top_n", default_map);
  FrcnnParam::test_rpn_post_nms_top_n = extract_int("test_rpn_post_nms_top_n", default_map);
  FrcnnParam::test_rpn_min_size = extract_float("test_rpn_min_size", default_map);

  // ========================================
  FrcnnParam::feat_stride = extract_int("feat_stride", default_map);
  FrcnnParam::anchors = extract_vector("anchors", default_map);
  FrcnnParam::n_classes = extract_int("n_classes", default_map);
}

void FrcnnParam::print_param(){

  LOG(INFO) << "== Frcnn Parameters ==";

  LOG(INFO) << "rpn_nms_thresh    : " << FrcnnParam::rpn_nms_thresh;
  LOG(INFO) << "rpn_pre_nms_top_n : " << FrcnnParam::rpn_pre_nms_top_n;
  LOG(INFO) << "rpn_post_nms_top_n: " << FrcnnParam::rpn_post_nms_top_n;
  LOG(INFO) << "rpn_min_size      : " << FrcnnParam::rpn_min_size;

  LOG(INFO) << "test_rpn_nms_thresh    : " << FrcnnParam::test_rpn_nms_thresh;
  LOG(INFO) << "test_rpn_pre_nms_top_n : " << FrcnnParam::test_rpn_pre_nms_top_n;
  LOG(INFO) << "test_rpn_post_nms_top_n: " << FrcnnParam::test_rpn_post_nms_top_n;
  LOG(INFO) << "test_rpn_min_size      : " << FrcnnParam::test_rpn_min_size;

  LOG(INFO) << "== Global Parameters ==";
  LOG(INFO) << "feat_stride          : " << FrcnnParam::feat_stride;
  LOG(INFO) << "anchors_size         : " << FrcnnParam::anchors.size();
  LOG(INFO) << "n_classes            : " << FrcnnParam::n_classes;
}

} // namespace detection
