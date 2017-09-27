#include <utility>
#include <iomanip>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/rpn_data_gen.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

void rangeFill(vector<int>& vec, int N) {
    vec.clear();
    for (int i = 0; i < N; ++i) vec.push_back(i);
}

bool operator== (const RoiIdx& idx1, const RoiIdx& idx2) {
    return idx1.h==idx2.h && idx1.w==idx2.w && idx1.k==idx2.k;
}

std::ostream& operator<< (std::ostream& os, const RoiIdx& idx) {
    os << VAR_STREAM(idx.h) << VAR_STREAM(idx.w) << VAR_STREAM(idx.k);
    return os;
}

//
// Box
//
bool operator< (const Box& box1, const Box& box2) {
    return box1.prob > box2.prob;
}

Box operator& (const Box& box1, const Box& box2) {
    Box ans;
    ans.x1 = _MAX(box1.x1, box2.x1);
    ans.y1 = _MAX(box1.y1, box2.y1);
    ans.x2 = _MIN(box1.x2, box2.x2);
    ans.y2 = _MIN(box1.y2, box2.y2);
    return ans;
}

std::ostream& operator<< (std::ostream& os, const Box& box) {
    os << std::fixed << std::setprecision(2)
       << box.x1 << " " 
       << box.y1 << " " 
       << box.x2 << " " 
       << box.y2;
    return os;
}

bool operator== (const Box& b1, const Box& b2) {
    return fabs(b1.x1-b2.x1)<FLOAT_EPS && 
        fabs(b1.y1-b2.y1)<FLOAT_EPS && 
        fabs(b1.x2-b2.x2)<FLOAT_EPS && 
        fabs(b1.y2-b2.y2)<FLOAT_EPS;
}

float boxIoU(const Box& box1, const Box& box2) {
    Box over = box1 & box2;
    return over.area() / (box1.area()+box2.area()-over.area());
}

std::ostream& operator<< (std::ostream& os, const BoxDelta& delta) {
    os << "Box(" 
       << delta.dx << ", " 
       << delta.dy << ", " 
       << delta.dw << ", " 
       << delta.dh << ")";
    return os;
}

bool operator== (const BoxDelta& db1, const BoxDelta& db2) {
    return fabs(db1.dx-db2.dx)<FLOAT_EPS && 
        fabs(db1.dy-db2.dy)<FLOAT_EPS && 
        fabs(db1.dw-db2.dw)<FLOAT_EPS && 
        fabs(db1.dh-db2.dh)<FLOAT_EPS;
}

std::ostream& operator<< (std::ostream& os, const RpnBatch& batch) {
    os << VAR_STREAM(batch.max_im_wid) << std::endl;
    os << VAR_STREAM(batch.max_im_hei) << std::endl;
    os << VAR_STREAM(batch.max_feat_wid) << std::endl;
    os << VAR_STREAM(batch.max_feat_hei) << std::endl;
    os << VAR_STREAM(batch.shuffle_idx) << std::endl;
    for (int i = 0; i < batch.image_inds.size(); ++i) {
        os << VAR_STREAM(batch.image_inds[i]) << std::endl;
    }
    return os;
}


//
// sample related
//
void ImageRois::rescale(float im_scale, map<int, pair<int, int> >& outmap) {
    CHECK(this->scale == 0) << "Image rois have been scaled.";
    this->scale = im_scale;
    s_hei = (int)round(hei*im_scale);
    s_wid = (int)round(wid*im_scale);
    feat_hei = outmap[s_hei].first;
    feat_wid = outmap[s_wid].second;
    CHECK(feat_hei>0) << "wrong map size for height:" << feat_hei;
    CHECK(feat_wid>0) << "wrong map size for width:" << feat_wid;
    
    for (int i = 0; i < boxes.size(); ++i) {
        boxes[i].x1 = std::max(0., round((boxes[i].x1+1.)*im_scale)-1.);
        boxes[i].y1 = std::max(0., round((boxes[i].y1+1.)*im_scale)-1.);
        boxes[i].x2 = std::max(0., round((boxes[i].x2+1.)*im_scale)-1.);
        boxes[i].y2 = std::max(0., round((boxes[i].y2+1.)*im_scale)-1.);
    }
}


//
// RpnDataGen
//

RpnDataGen::RpnDataGen(RpnDataParameter data_param, Phase phase) 
    : data_param_(data_param), phase_(phase) {
    int seed = data_param_.rand_seed();
    setup_rng_.reset(new Caffe::RNG(seed));

    int thread_num = 1;
    batch_num_ = data_param_.ims_per_batch() / thread_num;

    int dataset_num = data_param_.root_folder_size();
    for (int i=0; i<dataset_num; ++i) {
        LOG(INFO) << "-> folder[" << i << "]:" << data_param_.root_folder(i);
        root_folders_.push_back(data_param_.root_folder(i));
    }
}

void RpnDataGen::shuffleDataset() {
    CHECK(shuffle_inds_.size() == roisdb_.size());
    if (!data_param().debug()) {
        caffe::rng_t* rng = static_cast<caffe::rng_t*>(setup_rng_->generator());
        shuffle(shuffle_inds_.begin(), shuffle_inds_.end(), rng);
    }
}

void RpnDataGen::reserveData() {
    int thread_id = 0;
    int thread_num = 1;
    int all_rois_num = (int)roisdb_.size();
    vector<ImageRois> reserve_rois;
    for (int i=thread_id; i<all_rois_num; i += thread_num) {
        reserve_rois.push_back(roisdb_[i]);
    }
    roisdb_.swap(reserve_rois);
    rangeFill(shuffle_inds_, (int)roisdb_.size());
}

float RpnDataGen::calcScale(int wid, int hei) {
    float im_scale = (float)data_param().scale() / (float)_MIN( wid, hei );
    if( round( im_scale*_MAX(wid, hei) ) >  data_param().max_size()) {
        im_scale = (float)data_param().max_size() / (float)_MAX(wid, hei);
    }
    return im_scale;
}


void RpnDataGen::loadOutmap(map<int, pair<int, int> >& outmap) {
    string outmap_fn = data_param().source_outmap();
    std::ifstream ifs(outmap_fn.c_str());
    CHECK(ifs) << outmap_fn << " open error.";
    int size, w, h;
    while (ifs >> size >> w >> h) {
        outmap[size] = std::make_pair<int, int>(w, h);
    }
}

void RpnDataGen::loadAnchors() {
    string anchors_fn = data_param().anchors();
    std::ifstream ifs(anchors_fn.c_str());
    CHECK(ifs) << anchors_fn << " open error.";

    anchors_.clear();
    int num_anchors;
    ifs >> num_anchors;
    int x1, y1, x2, y2;
    while (ifs >> x1 >> y1 >> x2 >> y2) {
        anchors_.push_back(Box(x1, y1, x2, y2));
    }
    CHECK((int)anchors_.size() == num_anchors);
}

void RpnDataGen::nextBatch() {
    // select batch
    batch_.clear();
    for (int i = 0; i < batch_num_; ++i) {
        // select image
        int image_idx = shuffle_inds_[batch_.shuffle_idx++];
        const ImageRois& gt_rois = roisdb_[image_idx];

        // fill batch info
        batch_.image_inds.push_back(image_idx);
        batch_.max_im_hei = (gt_rois.s_hei > batch_.max_im_hei) ? gt_rois.s_hei : batch_.max_im_hei;
        batch_.max_im_wid = (gt_rois.s_wid > batch_.max_im_wid) ? gt_rois.s_wid : batch_.max_im_wid;
        batch_.max_feat_hei = (gt_rois.feat_hei > batch_.max_feat_hei) ? gt_rois.feat_hei : batch_.max_feat_hei;
        batch_.max_feat_wid = (gt_rois.feat_wid > batch_.max_feat_wid) ? gt_rois.feat_wid : batch_.max_feat_wid;
        
        if (batch_.shuffle_idx == (int)roisdb_.size()) {
            batch_.shuffle_idx = 0;
            shuffleDataset();
        }
    }
    // // ensure the blobs sizes are same if 
    // if (Caffe::getThreadNum() > 1) {
    //     batch_.max_im_hei = max_im_hei_;
    //     batch_.max_im_wid = max_im_wid_;
    //     batch_.max_feat_hei = max_feat_hei_;
    //     batch_.max_feat_wid = max_feat_wid_;
    // }
}

void RpnDataGen::loadRoisDb() {
    // read rois
    CHECK_EQ(data_param().source_size(), root_folders_.size());
    int num_dataset = data_param().source_size();
    roisdb_.clear();
    for (int i=0; i<num_dataset; ++i) {
        string rois_file = data_param().source(i);
        std::ifstream ifs(rois_file.c_str());
        CHECK(ifs) << rois_file << " open error.";
        LOG(INFO) << "loading dataset: " << data_param().source(i);
        string HASH;
        int image_id;
        int num_ds_rois = 0; // debug
        while (ifs >> HASH >> image_id) {
            CHECK(HASH=="#");
            ImageRois image_rois;
            int num_rois, num_ignore;
            CHECK(ifs >> image_rois.path >> image_rois.wid >> image_rois.hei >> num_rois >> num_ignore);
            CHECK(num_ignore == 0) << num_ignore << " ignore boxes.";
            image_rois.boxes.resize(num_rois);
            image_rois.dataset_idx = i;
            int _label;
            for (int i = 0; i < num_rois; ++i) {
                ifs >> _label 
                    >> image_rois.boxes[i].x1 
                    >> image_rois.boxes[i].y1 
                    >> image_rois.boxes[i].x2 
                    >> image_rois.boxes[i].y2;
            }
            // stock
            roisdb_.push_back(image_rois);
            num_ds_rois++;
        } // while
        LOG(INFO) << num_ds_rois << " samples loaded.";
    } // i

    // rescale rois
    rescaleRoisDb();

    // init inds
    rangeFill(shuffle_inds_, (int)roisdb_.size());
}

void RpnDataGen::rescaleRoisDb() {
    // load outmap
    map<int, pair<int, int> > outmap;
    loadOutmap(outmap);

    // rescale images and gt boxes
    for (int i = 0; i < (int)roisdb_.size(); ++i) {
        float im_scale = calcScale(roisdb_[i].wid, roisdb_[i].hei);
        roisdb_[i].rescale(im_scale, outmap);
    }

    // set max sizes
    max_im_hei_ = data_param().max_size();
    max_im_wid_ = data_param().max_size();
    max_feat_hei_ = outmap[max_im_hei_].second;
    max_feat_wid_ = outmap[max_im_wid_].first;
}

std::pair<int, int> RpnDataGen::mapPointToImage(int h, int w) {
    int cent_h, cent_w;
    switch (data_param().proposal_method()){
        case RpnDataParameter::PROPOSAL_STRIDE:
            cent_h = data_param().feat_stride_h()*(h+0.5);
            cent_w = data_param().feat_stride_w()*(w+0.5);
            break;
        case RpnDataParameter::PROPOSAL_RCPTFLD:
            NOT_IMPLEMENTED;
            break;
        default:
            LOG(FATAL) << "wrong proposal method.";
    }
    return std::make_pair<int, int>(cent_h, cent_w);
}

Box RpnDataGen::calcProposalBox(int h, int w, const Box& ank) {
    Box ans;
    std::pair<int, int> cent = mapPointToImage(h, w);
    int cent_h = cent.first;
    int cent_w = cent.second;
    // shift anchors 
    ans.x1 = ank.x1 + cent_w;
    ans.y1 = ank.y1 + cent_h;
    ans.x2 = ank.x2 + cent_w;
    ans.y2 = ank.y2 + cent_h;
    return ans;
}

float RpnDataGen::calcIoU(const Box& proposal, const Box& gt_box, const Box& rcpt_box, Box& target_box) {
    float iou = 0.0;
    target_box = gt_box;
    switch (data_param().iou_method()) {
        case RpnDataParameter::IOU_GT:
            break;
        case RpnDataParameter::IOU_GT_RCPT:
            target_box = gt_box & rcpt_box;
            break;
        case RpnDataParameter::IOU_GT_ANCHOR:
            target_box.x1 = _MAX(gt_box.x1, proposal.x1);
            target_box.x2 = _MIN(gt_box.x2, proposal.x2);
            break;
        default:
            LOG(FATAL) << "wrong IoU method.";
    }
    iou = boxIoU(proposal, target_box);
    return iou;
}

// NOTE: not consider the boundary situation
Box RpnDataGen::calcRcptfld(int h, int w) {
    int half_wid = (data_param().rcptfld_wid() - 1) / 2;
    int half_hei = (data_param().rcptfld_hei() - 1) / 2;
    Box ans = calcProposalBox(h, w, Box(-half_wid, -half_hei, half_wid, half_hei));
    return ans;
}

void RpnDataGen::genPNSamples(const ImageRois& gt_rois, ImagePNSamps& pn_samps) {
    // num_proposals = (H, W, K)
    const int H = gt_rois.feat_hei;
    const int W = gt_rois.feat_wid;
    const int K = anchors_num();
    int num_proposals = H*W*K;

    // preapre recorder to be used
    int num_gt = (int)gt_rois.boxes.size();
    float gt_max_iou[num_gt];
    int gt_max_cand[num_gt];
    float cand_max_iou[num_proposals];
    vector<bool> cand_boundary(num_proposals, false);
    vector<BoxDelta> cand_delta(num_proposals);
    memset(gt_max_iou,  0, sizeof(gt_max_iou));
    memset(gt_max_cand, 0, sizeof(gt_max_cand));
    memset(cand_max_iou,0, sizeof(cand_max_iou));

    CHECK(pn_samps.pos_samps.size() == 0);
    CHECK(pn_samps.pos_deltas.size() == 0);
    CHECK(pn_samps.neg_samps.size() == 0);

    // first loop to store max IoU
    for (int i = 0; i < num_proposals; ++i) {
        const int h = i/(W*K) % H;
        const int w = i/K  % W;
        const int k = i % K;
        CHECK(h>=0 && h<H && w>=0 && w<W && k>=0 && k<K);

        Box proposal = calcProposalBox(h, w, anchors(k));
        Box rcptfld = calcRcptfld(h, w);

        // check boundary cross
        if (proposal.x1<0 || proposal.y1<0 || proposal.x2>gt_rois.s_wid-1 || proposal.y2>gt_rois.s_hei-1) {
            cand_boundary[i] = true;
            continue;
        }

        for (int g = 0; g < gt_rois.boxes.size(); ++g) {
            Box target_box;
            float iou = calcIoU(proposal, gt_rois.boxes[g], rcptfld, target_box);
            // find max iou of a proposal
            if (iou > cand_max_iou[i]) {
                cand_max_iou[i] = iou;
                cand_delta[i].transform(proposal, target_box);
            }
            if (iou-1e-5 > gt_max_iou[g]) {
                gt_max_iou[g] = iou;
                gt_max_cand[g] = i;
            }
        }
    }

    // second loop to select samples
    vector<bool> select_by_gt(num_proposals, false);
    for (int g = 0; g < num_gt; ++g) {
        select_by_gt[gt_max_cand[g]] = true;
    }
    for (int i = 0; i < num_proposals; ++i) {
        if (cand_boundary[i]) continue;

        const int h = i/(W*K) % H;
        const int w = i/K  % W;
        const int k = i % K;
        CHECK(h>=0 && h<H && w>=0 && w<W && k>=0 && k<K);
        RoiIdx samp_idx(h, w, k);
        if (select_by_gt[i] || cand_max_iou[i] > data_param().fg_thresh()) {
            pn_samps.pos_samps.push_back(samp_idx);
            pn_samps.pos_deltas.push_back(cand_delta[i]);
        }else if(cand_max_iou[i] < data_param().bg_thresh()) {
            pn_samps.neg_samps.push_back(samp_idx);
        }
    }

    CHECK(pn_samps.pos_samps.size() == pn_samps.pos_deltas.size());
}

void RpnDataGen::calcMeanStd(const vector<ImagePNSamps>& pn_samps_vec) {
    float num_rois = 0.;
    float delta_sum[4], delta_squared_sum[4];
    memset(delta_sum, 0, sizeof(delta_sum));
    memset(delta_squared_sum, 0, sizeof(delta_squared_sum));
    for (int i = 0; i < (int)pn_samps_vec.size(); ++i) {
        for (int j = 0; j < (int)pn_samps_vec[i].pos_deltas.size(); ++j) {
            num_rois += 1.;
            for (int k = 0; k < 4; ++k) {
                delta_sum[k] += pn_samps_vec[i].pos_deltas[j][k];
                delta_squared_sum[k] += pow(pn_samps_vec[i].pos_deltas[j][k], 2.0);
            }
        }
    }

    // normalize 
    for (int i = 0; i < 4; ++i) {
        delta_means_[i] = delta_sum[i] / num_rois;
        delta_stds_[i]  = sqrt(delta_squared_sum[i]/num_rois - delta_means_[i]*delta_means_[i]);
    }

    // // write mean std to file
    // // TODO: get first id not use hard assigned 0
    // if (Caffe::getThreadId() == 0) {
    //     std::ofstream ofs(data_param().stds_means_file().c_str());
    //     ofs << std::fixed << std::setprecision(8) 
    //         << "STDS: " 
    //         << delta_stds_[0] << " " << delta_stds_[1] << " "
    //         << delta_stds_[2] << " " << delta_stds_[3] << std::endl
    //         << "MEANS: "
    //         << delta_means_[0] << " " << delta_means_[1] << " "
    //         << delta_means_[2] << " " << delta_means_[3] << std::endl;
    // }
}

void RpnDataGen::loadMeanStd() {
    std::ifstream ifs(data_param().stds_means_file().c_str());
    CHECK(ifs);
    string tag;
    ifs >> tag;
    CHECK(tag == "STDS:");
    for (int i = 0; i < 4; ++i) {
        ifs >> delta_stds_[i];
    }
    ifs >> tag;
    CHECK(tag == "MEANS:");
    for (int i = 0; i < 4; ++i) {
        ifs >> delta_means_[i];
    }
}

void RpnDataGen::prepareMeanStd() {
    switch (phase_) {
    case TRAIN:
        {
            // use random samples for calculating means/stds
            int mean_images_num = _MIN(data_param().mean_image_num(), (int)roisdb_.size());
            LOG(INFO) << "phase [TRAIN], calculating means and stds with " << mean_images_num << " images";
            CHECK(mean_images_num > 0);
            vector<ImagePNSamps> pn_samps_vec(mean_images_num);
            for (int i = 0; i < mean_images_num; ++i) {
                int image_idx = shuffle_inds_[i];
                const ImageRois& gt_rois = roisdb_[image_idx];
                genPNSamples(gt_rois, pn_samps_vec[i]);
            }
            calcMeanStd(pn_samps_vec);
        }
        break;
    case TEST:
        LOG(INFO) << "phase [TEST], loading means and stds";
        loadMeanStd();
        break;
    default: 
        LOG(FATAL) << "wrong phase.";
    }
    LOG(INFO) << "Means and Stds: ";
}

void RpnDataGen::prepareData() {
    // load anchors
    loadAnchors();

    // load gt rois
    loadRoisDb();
    
    // shuffle image indices
    shuffleDataset();

    // means and stds
    prepareMeanStd();

    // reserve data for current thread
    reserveData();

}

} // namespcae::caffe
