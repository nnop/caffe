#include <utility>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/rpn_data_gen.hpp"

using namespace std;

DEFINE_string(mode, "CPU", "CPU|GPU");
DEFINE_string(proto, "", "The test protobuf file.");
DEFINE_string(model, "", "The caffemodel file.");
DEFINE_double(scale, 0., "The scale.");
DEFINE_double(max_size, 0., "The max size.");
DEFINE_int32(stride_h, 0, "stride_h.");
DEFINE_int32(stride_w, 0, "stride_w.");
DEFINE_string(anchors, "", "The anchors file.");
DEFINE_string(std_mean, "", "The stds and means file.");
DEFINE_string(root_folder, "", "The image root folder.");
DEFINE_string(image_list, "", "The image list file.");
DEFINE_string(output, "", "File to save the predictions.");
DEFINE_double(conf_thresh, 0.9, "predict confidence thershold");
DEFINE_double(nms_thresh, 0.5, "nms threshold");
DEFINE_double(h_overlap_thresh, 0.7, "height overlap threshold");

namespace caffe {

void loadBoxes(string boxes_file, string& image_file, int& im_wid, int& im_hei, vector<Box>& boxes) {
    ifstream ifs(boxes_file.c_str());
    CHECK(ifs);

    string hashtag;
    int image_idx, _box_num, _ignore_num;
    ifs >> hashtag >> image_idx >> image_file;
    CHECK(ifs && hashtag=="#");
    ifs >> im_wid >> im_hei >> _box_num >> _ignore_num;

    float _p, x1, y1, x2, y2;
    while(ifs >> _p >> x1 >> y1 >> x2 >> y2) {
        boxes.push_back(Box(x1, y1, x2, y2));
    }
    CHECK(_box_num == boxes.size());
}

void dumpBoxes(ostream& os, string image_file, int image_idx, int im_wid, int im_hei, 
        const vector<Box>& boxes) {
    os << "# " << image_idx << std::endl;
    os << image_file << std::endl;
    os << im_wid << " " << im_hei << std::endl;
    os << (int)boxes.size() << " 0" << std::endl;
    for (int i=0; i<(int)boxes.size(); ++i) {
        os << std::fixed << std::setprecision(2) << boxes[i].prob << " " << boxes[i] << std::endl;
    }
}

/*****************
*  LinesExtractor  *
*****************/

class LinesExtractor {
    public:
        LinesExtractor(string boxes_file) {
            string image_file;
            int im_wid, im_hei;
            loadBoxes(boxes_file, image_file, im_wid, im_hei, boxes_);
            init();
        }
        LinesExtractor(const vector<Box>& boxes) : boxes_(boxes) {
            init();
        }
        void extractTextlines(vector<Box>& text_lines);

    private:
        void init() {
            boxes_num_ = (int)boxes_.size();
            adj_lists_.resize(boxes_num_);
        }
        void buildGraph();
        void mergeBoxes(vector<Box>& merge_boxes);
        bool isConnect(const Box& box1, const Box& box2);
        Box calcMergeBox(const vector<int>& conn_indices);
        void dfsLabel(int box_idx, int curr_label, vector<int>& box_labels);

    private:
        vector<Box> boxes_;
        int boxes_num_;
        vector<vector<int> > adj_lists_;
};

class TextPredictor {
    public:
        TextPredictor(string proto_file, string weights_file, string anchors_file, string means_stds_file,
                float scale, float max_size, int stride_h, int stride_w);
        string predictText(string root_folder, string image_file, int image_id=0);

    private:
        void init();
        float resizeImage(cv::Mat im_ori, cv::Mat& im);
        void loadAnchors(string anchors_file);
        void loadStdsMeans(string means_stds_file);
        Box calcProposal(int i, int j, int k);
        void nms(vector<Box>& cand_boxes, vector<Box>& nms_boxes);

    private:
        shared_ptr<Net<float> > net_;
        vector<Box> anchors_;
        float delta_means_[4], delta_stds_[4];
        float scale_, max_size_;
        int stride_h_, stride_w_;
};

/********************
*  Implimentation  *
********************/



TextPredictor::TextPredictor(string proto_file, string weights_file, string anchors_file, string stds_means_file,
        float scale, float max_size, int stride_h, int stride_w) 
        : scale_(scale), max_size_(max_size), stride_h_(stride_h), stride_w_(stride_w) {
    net_.reset(new Net<float>(proto_file, TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    LOG(INFO) << "load from: " << proto_file;
    LOG(INFO) << "weights: " << weights_file;

    loadAnchors(anchors_file);
    loadStdsMeans(stds_means_file);
    
    // norm bbox pred blobs
    shared_ptr<Layer<float> > bbox_pred_layer = net_->layer_by_name("proposal_bbox_pred");
    vector<shared_ptr<Blob<float> > >& bbox_pred_params = bbox_pred_layer->blobs();
    int n_delta_dim = bbox_pred_params[0]->num();
    int n_feat_dim = bbox_pred_params[0]->channels();
    CHECK_EQ(bbox_pred_params[0]->width(), 1);
    CHECK_EQ(bbox_pred_params[0]->height(), 1);

    float* pred_weights = bbox_pred_params[0]->mutable_cpu_data();
    float* pred_bias = bbox_pred_params[1]->mutable_cpu_data();
    for(int i=0; i<n_delta_dim; ++i) {
        for (int j=0; j<n_feat_dim; ++j) {
            float x = pred_weights[i*n_feat_dim+j];
            pred_weights[i*n_feat_dim+j] = x*delta_stds_[i%4];
        }
    }
    for(int i=0; i<n_delta_dim; ++i) {
        pred_bias[i] = pred_bias[i]*delta_stds_[i%4] + delta_means_[i%4];
    }
}

Box TextPredictor::calcProposal(int i, int j, int k) {
    int shift_x = (j+0.5)*stride_w_;
    int shift_y = (i+0.5)*stride_h_;
    Box ank = anchors_[k];
    Box prop_box;
    prop_box.x1 = ank.x1 + shift_x;
    prop_box.y1 = ank.y1 + shift_y;
    prop_box.x2 = ank.x2 + shift_x;
    prop_box.y2 = ank.y2 + shift_y;
    return prop_box;
}

void TextPredictor::loadAnchors(string anchors_file) {
    std::ifstream ifs(anchors_file.c_str());
    CHECK(ifs) << anchors_file << " open error.";

    anchors_.clear();
    int num_anchors;
    ifs >> num_anchors;
    int x1, y1, x2, y2;
    while (ifs >> x1 >> y1 >> x2 >> y2) {
        anchors_.push_back(Box(x1, y1, x2, y2));
    }
    CHECK((int)anchors_.size() == num_anchors);
    for (int i=0; i<num_anchors; ++i) {
        LOG(INFO) << "anchor[" << i << "]: " << anchors_[i];
    }
}

void TextPredictor::loadStdsMeans(string stds_means_file) {
    std::ifstream ifs(stds_means_file.c_str());
    CHECK(ifs);
    string HASH;
    ifs >> HASH >> delta_stds_[0] >> delta_stds_[1] >> delta_stds_[2] >> delta_stds_[3];
    CHECK(HASH == "STDS:");
    LOG(INFO) << "stds: " << BoxDelta(delta_stds_);
    ifs >> HASH >> delta_means_[0] >> delta_means_[1] >> delta_means_[2] >> delta_means_[3];
    CHECK(HASH == "MEANS:");
    LOG(INFO) << "means: " << BoxDelta(delta_means_);
}

float TextPredictor::resizeImage(cv::Mat im, cv::Mat &res) {
    int im_hei = im.rows;
    int im_wid = im.cols;

    float short_len = _MIN(im_hei, im_wid);
    float long_len = _MAX(im_hei, im_wid);

    float im_scale = scale_ / short_len;
    if(im_scale*long_len > max_size_) 
        im_scale = max_size_ / long_len;

    cv::resize(im, res, cv::Size(), im_scale, im_scale);
    return im_scale;
}

void TextPredictor::nms(vector<Box>& cand_boxes, vector<Box>& nms_boxes) {
    sort(cand_boxes.begin(), cand_boxes.end());
    int N = (int)cand_boxes.size();
    vector<bool> keep(N, true);
    for (int i=0; i<N; ++i) {
        for (int j=i+1; j<N; ++j) {
            if (boxIoU(cand_boxes[i], cand_boxes[j]) > FLAGS_nms_thresh)
                keep[j] = false;
        }
    }
    nms_boxes.clear();
    for (int i=0; i<N; ++i) {
        if (keep[i])
            nms_boxes.push_back(cand_boxes[i]);
    }
}

string TextPredictor::predictText(string root_folder, string image_file, int image_id) {
    ostringstream oss; 

    // load image, norm, resize
    CPUTimer timer, tot_timer;
    tot_timer.Start();
    timer.Start();
    LOG(INFO) << root_folder+image_file;
    cv::Mat im_ori = cv::imread(root_folder+image_file, 0);
    int image_w = im_ori.cols;
    int image_h = im_ori.rows;
    im_ori.convertTo(im_ori, CV_32F);
    cv::Mat im;
    float im_scale = resizeImage(im_ori, im);

    // fill input blobs
    vector<int> in_blob_shape(4);
    in_blob_shape[0] = 1; // N
    in_blob_shape[1] = 1; // IM_K
    in_blob_shape[2] = im.rows; // H
    in_blob_shape[3] = im.cols; // W
    const vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK(input_blobs.size()==1);
    input_blobs[0]->Reshape(in_blob_shape);
    for (int i=0; i<im.cols*im.rows; ++i) {
        input_blobs[0]->mutable_cpu_data()[i] = 143.0 - ((float*)im.data)[i];
    }
    float t_image = timer.MicroSeconds();
    
    // predict 
    timer.Start();
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    const float* bbox_pred = output_blobs[0]->cpu_data();
    const float* prob_pred = output_blobs[1]->cpu_data();
    float t_fwd = timer.MicroSeconds();

    int n_pred_bbox_dim = output_blobs[0]->channels();
    CHECK(n_pred_bbox_dim % 4 == 0);
    int K = n_pred_bbox_dim / 4;
    int H = output_blobs[0]->height();
    int W = output_blobs[0]->width();

    // valid
    timer.Start();
    vector<Box> cand_boxes;
    for (int k=0; k<K; ++k) {
        for (int i=0; i<H; ++i) {
            for (int j=0; j<W; ++j) {
                float p = prob_pred[(K+k)*H*W+i*W+j];
                if(p < FLAGS_conf_thresh) continue;
                // delta
                BoxDelta dlt;
                dlt[0] = bbox_pred[(k*4+0)*H*W + i*W + j];
                dlt[1] = bbox_pred[(k*4+1)*H*W + i*W + j];
                dlt[2] = bbox_pred[(k*4+2)*H*W + i*W + j];
                dlt[3] = bbox_pred[(k*4+3)*H*W + i*W + j];
                // proposal
                Box prop_box = calcProposal(i, j, k);
                Box pred_box = dlt.inv_transform(prop_box);
                pred_box.prob = p;
                cand_boxes.push_back(pred_box);
            }
        }
    }
    float t_valid = timer.MicroSeconds();

    // nms
    timer.Start();
    vector<Box> nms_boxes;
    nms(cand_boxes, nms_boxes);
    float t_nms = timer.MicroSeconds();

    // merge
    timer.Start();
    LinesExtractor lines_ext(nms_boxes);
    vector<Box> lines;
    lines_ext.extractTextlines(lines);
    float t_merge = timer.MicroSeconds();

    // scale back
    for (int i=0; i<lines.size(); ++i) {
        lines[i].rescale(1.0 / im_scale);
    }
    dumpBoxes(oss, image_file, image_id, image_w, image_h, lines);

    LOG(INFO) << image_id << " --> " << image_file << " "<< im.size();
    LOG(INFO) << "data prepare: " << t_image / 1000. << " ms";
    LOG(INFO) << "forward time: " << t_fwd / 1000. << " ms";
    LOG(INFO) << "  valid time: " << cand_boxes.size() << " / " << t_valid / 1000. << " ms";
    LOG(INFO) << "    nms time: " << nms_boxes.size() << " / " << t_nms / 1000. << " ms";
    LOG(INFO) << "  merge time: " << lines.size() << " / " << t_merge / 1000. << " ms";
    LOG(INFO) << "  total time: " << tot_timer.MicroSeconds() / 1000. << " ms";

    return oss.str();
}


/************************************
*  LinesExtractor implementations  *
************************************/

bool LinesExtractor::isConnect(const Box& box1, const Box& box2) {
    // not overlap 
    Box inter_box = box1 & box2;
    if (inter_box.area() == 0) return false;

    // in another box
    if (box1.in(box2) || box2.in(box1))
        return true;

    // overlap horizontally
    if (inter_box.height() / _MAX(box1.height(), box2.height()) < FLAGS_h_overlap_thresh)
        return false;
    else
        return true;
}



void LinesExtractor::buildGraph() {
    for (int i=0; i<boxes_num_; ++i ) {
        for (int j=i+1; j<boxes_num_; ++j ) {
            if (isConnect(boxes_[i], boxes_[j])) {
                adj_lists_[i].push_back(j);
                adj_lists_[j].push_back(i);
            }
        }
    }
}

void LinesExtractor::extractTextlines(vector<Box>& text_lines) {
    buildGraph();
    mergeBoxes(text_lines);
}

void LinesExtractor::dfsLabel(int box_idx, int curr_label, vector<int>& box_labels) {
    // return if labeled
    if (box_labels[box_idx]!=-1)
        return; 

    box_labels[box_idx] = curr_label;
    for (int i=0; i<(int)adj_lists_[box_idx].size(); ++i) {
        dfsLabel(adj_lists_[box_idx][i], curr_label, box_labels);
    }
}

Box LinesExtractor::calcMergeBox(const vector<int>& conn_indices) {
    float min_x1 = 1e5, max_x2 = -1e5, mean_y1 = 0., mean_y2 = 0.;
    for (int i=0; i<(int)conn_indices.size(); ++i) {
        int box_idx = conn_indices[i];
        if (boxes_[box_idx].x1 < min_x1) min_x1 = boxes_[box_idx].x1;
        if (boxes_[box_idx].x2 > max_x2) max_x2 = boxes_[box_idx].x2;
        mean_y1 += boxes_[box_idx].y1;
        mean_y2 += boxes_[box_idx].y2;
    }
    mean_y1 /= (float)conn_indices.size();
    mean_y2 /= (float)conn_indices.size();

    return Box(min_x1, mean_y1, max_x2, mean_y2);
}

void LinesExtractor::mergeBoxes(vector<Box>& merge_boxes) {  
    // calc connected component with DFS
    //  -1 means non visited
    vector<int> boxes_labels(boxes_num_, -1);
    int curr_label = -1;
    for (int i=0; i<boxes_num_; ++i) {
        // non visited
        if (boxes_labels[i] == -1) {
            dfsLabel(i, ++curr_label, boxes_labels);
        }
    }

    // calc merged boxes
    int comps_num = curr_label+1;
    vector<vector<int> > components(comps_num);
    for (int i=0; i<boxes_num_; ++i) 
        components[boxes_labels[i]].push_back(i);
    CHECK((int)merge_boxes.size() == 0);
    for (int i=0; i<comps_num; ++i) {
        Box tmp_box = calcMergeBox(components[i]);
        merge_boxes.push_back(tmp_box);
    }
    LOG(INFO) << "text lines: " << comps_num;
}


} // caffe

int main(int argc, char *argv[])
{
    caffe::GlobalInit(&argc, &argv);
    CHECK_GT(FLAGS_proto.size(), 0);
    CHECK_GT(FLAGS_model.size(), 0);
    CHECK_GT(FLAGS_anchors.size(), 0);
    CHECK_GT(FLAGS_std_mean.size(), 0);
    CHECK_GT(FLAGS_image_list.size(), 0);
    CHECK_GT(FLAGS_output.size(), 0);
    CHECK_NE(FLAGS_scale, 0.);
    CHECK_NE(FLAGS_max_size, 0.);
    CHECK_NE(FLAGS_stride_h, 0);
    CHECK_NE(FLAGS_stride_w, 0);

    CHECK_EQ(argc, 1) << argc;

    if (FLAGS_mode == "CPU")
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
    else if (FLAGS_mode == "GPU")
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
    else {
      std::cout << "Wrong caffe mode, only CPU|GPU support." << std::endl;
      return -1;
    }
    caffe::Caffe::SetDevice(0);
    FLAGS_logtostderr = 1;

    caffe::TextPredictor tp(FLAGS_proto, FLAGS_model, FLAGS_anchors, FLAGS_std_mean,
            FLAGS_scale, FLAGS_max_size, FLAGS_stride_h, FLAGS_stride_w);

    ifstream ifs(FLAGS_image_list.c_str());
    ofstream ofs(FLAGS_output.c_str());
    string image_file;
    int image_id = 0;
    while (ifs >> image_file) {
        string out_str = tp.predictText(FLAGS_root_folder, image_file, image_id++);
        ofs << out_str;
    }

    return 0;
}
