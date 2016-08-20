% RPN and FCN test and combine them with multiple boxes on ilsvrc
% 
% saved separately in each image and will transfer to Kun Wang
% refactor by hyli on August 11, 2016
% ---------------------------------------------------------
caffe.reset_all();
clear; run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');

% use ID = F15, RPN + FCN, given attractioNet, ss and edgebox
% ===========================================================
% ======================= USER DEFINE =======================
% change to point to your devkit install
root_path = './datasets/ilsvrc14_det';
opts.gpu_id = 0;            % single-gpu version, index from 0

% all 'test' model, so dont be surprised that initial roidb are zeros!
% which means there's no GT in these fucking datasets.
%which_dataset = 'real_test';
%which_dataset = 'val1_14';
%which_dataset = 'val1_13';
which_dataset = 'pos1k_13';

external_box_list{1} = 'ss';
external_box_list{2} = 'edge_nms0_5';
external_box_list{3} = 'attract_nms0_65';
load_name = cell(length(external_box_list), 1);
for i = 1:3
    load_name{i} = sprintf('./box_proposals/%s/boxes_%s.mat', ...
        which_dataset, external_box_list{i});   
    % TODO: check file exist. assert('');
    assert(exist(load_name{i}, 'file')==2, ...
        sprintf('fuck! file does not exist! (%s)', load_name{i}));
end
dataset = [];
dataset = Dataset.ilsvrc14(dataset, which_dataset, false, root_path);
% ===================== USER DEFINE END =====================
% ===========================================================
model = Model.VGG16_for_Faster_RCNN(...
    'solver_10w30w_ilsvrc_9anchor', 'test_9anchor', ...     % rpn
    'solver_5w15w_2', 'test_2' ...                          % fast_rcnn
    );
% --------------------------- FCN ----------------------------
fast_rcnn_net_file = [{'train14'}, {'final'}];
binary_train                = true;
cache_base_FCN              = 'F02_ls149';
share_data_FCN              = '';
fcn_fg_thresh               = 0.5;
fcn_bg_thresh_hi            = 0.5;
fcn_bg_thresh_lo            = 0.1;
fcn_scales                  = 600;
fcn_fg_fraction             = 0.25;
fcn_max_size                = 1000;
fast_rcnn_after_nms_topN    = 2000;
fast_nms_overlap_thres      = 0.65;
% --------------------------- RPN ----------------------------
cache_base_RPN              = 'M02_s31';
skip_rpn_test               = false;
update_roi                  = true;
update_roi_name             = 'rpn';    % change here in new version
detect_exist_config_file    = true;
model.stage1_rpn.nms.note   = '0.6';   
model.stage1_rpn.nms.nms_overlap_thres = 0.6;
% ==========================================================
% ==========================================================
model.stage1_rpn.nms.mult_thr_nms = false;
if isnan(str2double(model.stage1_rpn.nms.note)), model.stage1_rpn.nms.mult_thr_nms = true; end
model = Faster_RCNN_Train.set_cache_folder(cache_base_RPN, cache_base_FCN, model);

caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

[conf_proposal, conf_fast_rcnn] = Faster_RCNN_Train.set_config( ...
    cache_base_RPN, model, detect_exist_config_file );
conf_proposal.cache_base_proposal = cache_base_RPN;
fg_thresh = 0.5;        % 0.7 default
bg_thresh_hi = 0.5;     % 0.3 default
scales = 600;
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

if 0
%% step 1, RPN stage 1 extraction
cprintf('blue', '\nStage one proposal TEST on val data ...\n');
dataset.roidb_test = RPN_TEST_ilsvrc_hyli(...
    'train14', 'final', model, ...
    dataset.imdb_test, dataset.roidb_test, conf_proposal, ...
    'update_roi',           update_roi, ...
    'update_roi_name',      update_roi_name, ...
    'skip_rpn_test',        skip_rpn_test, ...
    'gpu_id',               opts.gpu_id ...
    );
end
%% step 2, add more proposal here
name = sprintf('comboALL');
FLIP = 'unflip';
new_roidb_file = fullfile(pwd, 'imdb/cache/ilsvrc', ...
    ['roidb_' dataset.roidb_test.name '_' FLIP sprintf('_%s.mat', name)]);
test_sub_folder_suffix = 'F15c';
keep_raw = true;

if ~exist(new_roidb_file, 'file')
    cprintf('blue', 'append external boxes to newly-generated roidb...\n');
    for i = 1:length(load_name)
        ld = load(load_name{i});
        try aboxes = ld.aboxes; catch, aboxes = ld.boxes_uncut; end
        roidb_regions = [];
        roidb_regions.boxes = aboxes;
        roidb_regions.images = dataset.imdb_test.image_ids;
        % update roidb in 'imdb' folder
        roidb_from_proposal(dataset.imdb_test, dataset.roidb_test, ...
            roidb_regions, 'keep_raw_proposal', keep_raw, 'mat_file', new_roidb_file);
        % update the roidb in matlab dynamically
        ld = load(new_roidb_file);
        dataset.roidb_test.rois = ld.rois;
    end
else
    cprintf('blue', 'directly load all (%d + rpn) results...\n', length(load_name));
    ld = load(new_roidb_file);
    dataset.roidb_test.rois = ld.rois;
end

%% step 3, merge all result and get FCN output
cprintf('blue', '\nStage two Fast-RCNN cascade TEST...\n');
% if adding more proposals, you need to increase the number here
test_max_per_image          = 30000; 
test_avg_per_image          = test_max_per_image;

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
