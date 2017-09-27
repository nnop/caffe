#ifndef RPN_DATA_GEN_H
#define RPN_DATA_GEN_H

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// for debug
#define VAR_STREAM(x) #x"=" << x << ", "

#define _MAX(A,B) ((A)>(B)?(A):(B))
#define _MIN(A,B) ((A)<(B)?(A):(B))

#define FLOAT_EPS 1e-5

void rangeFill(vector<int>& vec, int N);

//
// Box related
//  currently, we don't consider label
//
struct Box {
    float x1, y1, x2, y2;
    float prob;

    explicit Box(float _x1=0.0, float _y1=0.0, float _x2=0.0, float _y2=0.0, float _p=1.0) :
        x1(_x1), y1(_y1), x2(_x2), y2(_y2), prob(_p) {}

    float width()  const { return _MAX(x2-x1+1, 0.); }
    float height() const { return _MAX(y2-y1+1, 0.); }
    float area()   const { return width()*height(); }
    std::pair<int, int> center() const {
        int cent_x = (int)round(float(x1+x2)/2.);
        int cent_y = (int)round(float(y1+y2)/2.);
        return make_pair<int, int>(cent_x, cent_y);
    }
    
    // for read-only access
    float operator[] (int i) const {
        switch (i) {
            case 0: return x1;
            case 1: return y1;
            case 2: return x2;
            case 3: return y2;
            default: LOG(FATAL) << "wrong index.";
        }
    }
    bool in(const Box& b) const {
        return x1>=b.x1 && y1>=b.y1 && x2<=b.x2 && y2<=b.y2;
    }
    void rescale(float s) {
        x1 = x1*s;
        y1 = y1*s;
        x2 = x2*s;
        y2 = y2*s;
    }
};


bool operator< (const Box& box1, const Box& box2);
Box operator& (const Box& box1, const Box& box2);
bool operator== (const Box& box1, const Box& box2);
std::ostream& operator<< (std::ostream& os, const Box& box);
float boxIoU(const Box& box1, const Box& box2);

// delta struct
struct BoxDelta{
    float dx, dy, dw, dh;
    explicit BoxDelta(float _dx=0.0, float _dy=0.0, float _dw=0.0, float _dh=0.0) 
        : dx(_dx), dy(_dy), dw(_dw), dh(_dh) {}
    BoxDelta(float dlt_arr[]) 
        : dx(dlt_arr[0]), dy(dlt_arr[1]), dw(dlt_arr[2]), dh(dlt_arr[3]) {}

    void transform(const Box& prop_box, const Box& tgt_box) {
        float prop_wid = prop_box.width();
        float prop_hei = prop_box.height();
        float prop_cent_x = prop_box.center().first;
        float prop_cent_y = prop_box.center().second;

        float tgt_wid = tgt_box.width();
        float tgt_hei = tgt_box.height();
        float tgt_cent_x = tgt_box.center().first;
        float tgt_cent_y = tgt_box.center().second;

        dx = (tgt_cent_x - prop_cent_x) / prop_wid;
        dy = (tgt_cent_y - prop_cent_y) / prop_hei;
        dw = std::log(tgt_wid / prop_wid);
        dh = std::log(tgt_hei / prop_hei);
    }

    Box inv_transform(const Box& prop_box) {
        float prop_w = prop_box.x2 - prop_box.x1 + 1;
        float prop_h = prop_box.y2 - prop_box.y1 + 1;
        float prop_cent_x = (prop_box.x1 + prop_box.x2) / 2.;
        float prop_cent_y = (prop_box.y1 + prop_box.y2) / 2.;
        float pred_cent_x = dx*prop_w + prop_cent_x;
        float pred_cent_y = dy*prop_h + prop_cent_y;
        float pred_w = exp(dw)*prop_w;
        float pred_h = exp(dh)*prop_h;
        float x1 = pred_cent_x - 0.5*pred_w;
        float x2 = pred_cent_x + 0.5*pred_w;
        float y1 = pred_cent_y - 0.5*pred_h;
        float y2 = pred_cent_y + 0.5*pred_h;
        return Box(x1, y1, x2, y2);
    }

    // used only for access
    float operator[] (int i) const {
        switch (i) {
            case 0: return dx;
            case 1: return dy;
            case 2: return dw;
            case 3: return dh;
            default: LOG(FATAL) << "wrong index.";
        }
    }
    float& operator[] (int i) {
        switch (i) {
            case 0: return dx;
            case 1: return dy;
            case 2: return dw;
            case 3: return dh;
            default: LOG(FATAL) << "wrong index.";
        }
    }
};

bool operator== (const BoxDelta& delta1, const BoxDelta& delta2);
std::ostream& operator<< (std::ostream& os, const BoxDelta& delta);

// show pair
template <typename T1, typename T2>
std::ostream& operator<< (std::ostream& os, const std::pair<T1, T2>& p) {
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}

//
// sample related
//
struct RoiIdx {
    int h, w, k;
    explicit RoiIdx(int _h=-1, int _w=-1, int _k=-1) : 
        h(_h), w(_w), k(_k) {}
};

std::ostream& operator<< (std::ostream& os, const RoiIdx& idx);
typedef vector<RoiIdx>::const_iterator SampIter;

bool operator== (const RoiIdx& idx1, const RoiIdx& idx2);

struct ImageRois {
    string path;
    int dataset_idx;
    int hei, wid;
    int s_hei, s_wid;
    int feat_hei, feat_wid;
    vector<Box> boxes;
    float scale;

    ImageRois() : dataset_idx(0), hei(0), wid(0), s_hei(0), s_wid(0), scale(0.) {}
    // rescale image size and gt boxes
    void rescale(float im_scale, map<int, pair<int, int> >& outmap);
};

struct ImagePNSamps {
    vector<RoiIdx> pos_samps;
    vector<BoxDelta> pos_deltas;
    vector<RoiIdx> neg_samps;
};

//
// RpnBatch
//
struct RpnBatch {
    int max_im_wid;
    int max_im_hei;
    int max_feat_wid;
    int max_feat_hei;
    int shuffle_idx;
    vector<int> image_inds;
    RpnBatch() : max_im_wid(0), max_im_hei(0), 
        max_feat_wid(0), max_feat_hei(0), shuffle_idx(0) {}
    void clear() {
        max_im_wid = 0;
        max_im_hei = 0;
        max_feat_wid = 0;
        max_feat_hei = 0;
        image_inds.clear();
    }
};

std::ostream& operator<< (std::ostream& os, const RpnBatch& batch);

//
// RpnDataGen
//
class RpnDataGen {
public:
    RpnDataGen(RpnDataParameter data_param, Phase phase=TRAIN);
    virtual ~RpnDataGen() {};

    //
    // use cases
    //
    // build the whole sample database
    void prepareData();
    // select images in next batch
    void nextBatch();
    // shuffle
    void shuffleDataset();
    // reserve data for only current thread
    void reserveData();
    
    // load anchors
    void loadAnchors();
    // calc image centers of map point (h, w)
    std::pair<int, int> mapPointToImage(int h, int w);
    // outmap, size: (hei, wid)
    void loadOutmap(map<int, pair<int, int> >& outmap);
    // read rois file
    void loadRoisDb();
    // rescale roisdb
    void rescaleRoisDb();
    // compute the proposal box given a position
    Box calcProposalBox(int h, int w, const Box& ank);
    // compute scale for an image
    float calcScale(int wid, int hei);
    // calc IoU
    float calcIoU(const Box& proposal, const Box& gt_box, const Box& rcpt_box, Box& target_box);
    // calc receptive field of (h, w)
    Box calcRcptfld(int h, int w);
    void prepareMeanStd();
    // compute mean/std
    void calcMeanStd(const vector<ImagePNSamps>& pn_samps_vec);
    // load mean/std from file
    void loadMeanStd();
    // generate samples
    void genPNSamples(const ImageRois& gt_rois, ImagePNSamps& pn_samps);

    //
    // getters
    //
    int batch_size() { return data_param_.batch_size(); }
    int channel_num() { return 1; }
    int anchors_num() { return (int)anchors_.size(); }
    string root_folder(int i) { return data_param_.root_folder(i); }
    float bg_weight() { return data_param_.bg_weight(); }
    float mean() { return data_param_.mean_value(); }
    const Box& anchors(int i) { 
        CHECK(i<anchors_.size() && i>=0) << VAR_STREAM(anchors_.size()) << VAR_STREAM(i);
        return anchors_[i]; 
    }
    const RpnBatch& currBatch() const { return batch_; }
    const RpnDataParameter& data_param() const { return data_param_; }

    //
    // member vars
    //
    RpnDataParameter data_param_;
    RpnBatch batch_;    // current batch
    vector<Box> anchors_;
    vector<int> shuffle_inds_;
    vector<ImageRois> roisdb_;
    vector<string> root_folders_;
    float delta_means_[4], delta_stds_[4];
    Phase phase_;
    shared_ptr<Caffe::RNG> setup_rng_;
    int batch_num_;
    int max_im_hei_, max_im_wid_;
    int max_feat_hei_, max_feat_wid_;
};

} /* caffe */ 

#endif /* RPN_DATA_GEN_H */
