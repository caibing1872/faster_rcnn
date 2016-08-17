% RPN and FCNtraining and testing on ilsvrc
% refactor by hyli on July 28, 2016
% ---------------------------------------------------------
caffe.reset_all();
clear; run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');
opts.do_val = true;
% ===========================================================
% ======================= USER DEFINE =======================
use_flipped = false;
opts.gpu_id = 0;
opts.train_key = 'train14';
% load paramters from the 'models' folder
model = Model.VGG16_for_Faster_RCNN(...
    'solver_10w30w_ilsvrc_9anchor', 'test_9anchor', ...     % rpn
    'solver_5w15w_2', 'test_2' ...                          % fast_rcnn
    );
% --------------------------- FCN ----------------------------
fast_rcnn_net_file = [{'train14'}, {'final'}];
% if you want to generate new train_val_data, 'update_roi' must be set
% true; otherwise you can set it false to directly use existing data.
% update: you MUST update roi when test (TODO: explain more here).
update_roi                  = true;
% name in the imdb folder after adding NMS additional boxes
update_roi_name             = '1';
% update_roi_name             = 'M27_nms0.55';

binary_train                = true;
% FCN cache folder name
% cache_base_FCN              = 'FCN_try_local';
cache_base_FCN              = 'F02_ls149';
share_data_FCN              = '';
fcn_fg_thresh               = 0.5;
fcn_bg_thresh_hi            = 0.5;
fcn_bg_thresh_lo            = 0.1;
fcn_scales                  = [600];
fcn_fg_fraction             = 0.25;
fcn_max_size                = 1000;

fast_rcnn_after_nms_topN    = 2000;
fast_nms_overlap_thres = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4];

% --------------------------- RPN ----------------------------
% NOTE: this variable stores BOTH RPN and FCN in the 'config_temp' folder
% cache_base_RPN = 'NEW_ILSVRC_ls139';
cache_base_RPN = 'M02_s31';
% share_data_RPN = 'M04_ls149';
share_data_RPN = '';
% won't do test and compute recall
skip_rpn_test               = false;

model.anchor_size = 2.^(3:5);
model.ratios = [0.5, 1, 2];
detect_exist_config_file    = true;
detect_exist_train_file     = true;
% model.stage1_rpn.nms.note = '0.55';   % must be a string
% model.stage1_rpn.nms.nms_overlap_thres = 0.55;

% model.stage1_rpn.nms.note = 'multiNMS_1a';   % must be a string
% default
% model.stage1_rpn.nms.nms_iou_thrs   = [0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.60, 0.55];
% model.stage1_rpn.nms.max_per_image  = [2000, 1000,  400,  200,  100,   40,   20,   10];
fg_thresh = 0.5;        % 0.7 default
bg_thresh_hi = 0.5;     % 0.3 default
scales = [600];
% ==========================================================
% ==========================================================

model.stage1_rpn.nms.mult_thr_nms = false;
if isnan(str2double(model.stage1_rpn.nms.note)), model.stage1_rpn.nms.mult_thr_nms = true; end
model = Faster_RCNN_Train.set_cache_folder(cache_base_RPN, cache_base_FCN, model);
% finetune
if exist('ft_file', 'var')
    net_file = ft_file;
    fprintf('\ninit from another model\n');
else
    net_file = model.stage1_rpn.init_net_file;
end

caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();
% config, must be input after setting caffe
% in the 'proposal_config.m' file
% TODO change the saving mechanism here
[conf_proposal, conf_fast_rcnn] = Faster_RCNN_Train.set_config( ...
    cache_base_RPN, model, detect_exist_config_file );
conf_proposal.cache_base_proposal = cache_base_RPN;
conf_proposal.fg_thresh = fg_thresh;
conf_proposal.bg_thresh_hi = bg_thresh_hi;
conf_proposal.scales = scales;

if isempty(share_data_FCN)
    conf_fast_rcnn.data_name = cache_base_FCN;
else
    conf_fast_rcnn.data_name = share_data_FCN;
end

conf_fast_rcnn.fcn_fg_thresh        = fcn_fg_thresh;
conf_fast_rcnn.fcn_bg_thresh_hi     = fcn_bg_thresh_hi;
conf_fast_rcnn.fcn_bg_thresh_lo     = fcn_bg_thresh_lo;
conf_fast_rcnn.fcn_scales           = fcn_scales;
conf_fast_rcnn.fcn_fg_fraction      = fcn_fg_fraction;
conf_fast_rcnn.fcn_max_size         = fcn_max_size;
conf_fast_rcnn.update_roi_name      = update_roi_name;

% train/test data
% init:
%   imdb_train, roidb_train, cell;
%   imdb_test, roidb_test, struct
dataset = [];
% change to point to your devkit install
root_path = './datasets/ilsvrc14_det';
% dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);
dataset = Dataset.ilsvrc14(dataset, 'test_no_gt', false, root_path);

%%
% test: compute recall and update roidb on TEST
cprintf('blue', '\nStage one proposal TEST on val data ...\n');
dataset.roidb_test = RPN_TEST_ilsvrc_hyli(...
    'train14', 'final', model, ...
    dataset.imdb_test, dataset.roidb_test, conf_proposal, ...
    'mult_thr_nms',         model.stage1_rpn.nms.mult_thr_nms, ...
    'nms_iou_thrs',         model.stage1_rpn.nms.nms_iou_thrs, ...
    'max_per_image',        model.stage1_rpn.nms.max_per_image, ...
    'update_roi',           update_roi, ...
    'update_roi_name',      update_roi_name, ...
    'skip_rpn_test',        skip_rpn_test, ...
    'gpu_id',               opts.gpu_id ...
    );

% add more proposal here
% if adding more proposals, you need to increase the number here
test_max_per_image          = 10001; %10000; %100;
% if avg == max_per_im, there's no reduce in the number of boxes.
test_avg_per_image          = 10001; %10000; %500; %40;

name = 'combo_1_with_rpn';
FLIP = 'unflip';
new_roidb_file = fullfile(pwd, 'imdb/cache/ilsvrc', ...
    ['roidb_' dataset.roidb_test.name '_' FLIP sprintf('_%s.mat', name)]);
<<<<<<< HEAD
test_sub_folder_suffix = 'F15a';
keep_raw = true;
=======
test_sub_folder_suffix = 'F15b';
keep_raw = false;
>>>>>>> 85d5808b382b9c0c7a6361104eec3167865f1ea4

load_name{1} = './box_proposals/val2/aug_1st_ss/boxes_right_format.mat';
load_name{2} = './box_proposals/val2/aug_1st_edge/boxes_nms_0.50.mat';
load_name{3} = './box_proposals/val2/attractioNet/boxes_nms_0.65.mat';

if ~exist(new_roidb_file, 'file')
    
    for i = 1:length(load_name)
        % load attention boxes
        ld = load(load_name{i});
        try aboxes = ld.aboxes; catch, aboxes = ld.boxes_uncut; end
        roidb_regions = [];
        roidb_regions.boxes = aboxes;
        roidb_regions.images = dataset.imdb_test.image_ids;
        % update roidb in 'imdb' folder
        roidb_from_proposal(dataset.imdb_test, dataset.roidb_test, ...
            roidb_regions, 'keep_raw_proposal', keep_raw, 'mat_file', new_roidb_file);
        if i == 1, keep_raw = true; end
    end
end
ld = load(new_roidb_file);
dataset.roidb_test.rois = ld.rois;

% 'net_file', model.stage1_fast_rcnn.output_model_file, ...
cprintf('blue', '\nStage two Fast-RCNN cascade TEST...\n');
fast_rcnn_test(conf_fast_rcnn, dataset.imdb_test, dataset.roidb_test, ...
    'net_def_file',             model.stage1_fast_rcnn.test_net_def_file, ...
    'net_file',                 fast_rcnn_net_file, ...
    'cache_name',               model.stage1_fast_rcnn.cache_name, ...
    'binary',                   binary_train, ...
    'max_per_image',            test_max_per_image, ...
    'avg_per_image',            test_avg_per_image, ...
    'nms_overlap_thres',        fast_nms_overlap_thres, ...
    'test_sub_folder_suffix',   test_sub_folder_suffix, ...
    'after_nms_topN',           fast_rcnn_after_nms_topN ...
    );

exit;