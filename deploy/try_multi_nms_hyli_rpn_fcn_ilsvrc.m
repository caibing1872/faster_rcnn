% RPN and FCN test on ilsvrc, multi-thres NMS, only use cpu (if boxes are saved)
% but still, the gpu is occupied for initiating Caffe.
% refactor by hyli on Aug 2nd, 2016
% ---------------------------------------------------------
%caffe.reset_all();
clear; run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');
opts.do_val = true;
% ===========================================================
% ======================= USER DEFINE =======================
opts.gpu_id = 1;
opts.train_key = 'train14';

% model
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
cache_base_FCN              = 'F02_ls149';
share_data_FCN              = 'F04_ls149';
% fcn_fg_thresh               = 0.5;
% fcn_bg_thresh_hi            = 0.5;
% fcn_bg_thresh_lo            = 0.1;
% fcn_scales                  = [600];
% fcn_fg_fraction             = 0.25;
% fcn_max_size                = 1000;

% test NMS configuration
test_max_per_image          = 2000; %1000; %100;
% if avg == max_per_im, there's no reduce in the number of boxes.
test_avg_per_image          = 2000; %1000; %500; %40;
fast_rcnn_after_nms_topN    = 2000;
fast_nms_overlap_thres = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5];


% -------------------------------------------------------------
% --------------------------- RPN ----------------------------
% NOTE: this variable stores BOTH RPN and FCN in the 'config_temp' folder
% cache_base_RPN = 'NEW_ILSVRC_ls139';
cache_base_RPN = 'M02_s31';
% share_data_RPN = 'M04_ls149';
share_data_RPN = '';

skip_rpn_test               = false;     % won't do test and compute recall
% fg_thresh = 0.5;        % 0.7 default
% bg_thresh_hi = 0.5;     % 0.3 default
% scales = [600];

model.stage1_rpn.nms.mult_thr_nms = true;
model = Faster_RCNN_Train.set_cache_folder(cache_base_RPN, cache_base_FCN, model);

%caffe.set_device(opts.gpu_id);
%caffe.set_mode_gpu();
save_intermediate_box = 'raw_boxes_before_nms_M02_unflip.mat';
% config, must be input after setting caffe
% in the 'proposal_config.m' file
% TODO change the saving mechanism here
[conf_proposal, conf_fast_rcnn] = Faster_RCNN_Train.set_config( ...
    cache_base_RPN, model, true );
conf_proposal.cache_base_proposal = cache_base_RPN;
% conf_proposal.fg_thresh = fg_thresh;
% conf_proposal.bg_thresh_hi = bg_thresh_hi;
% conf_proposal.scales = scales;
if isempty(share_data_FCN)
    conf_fast_rcnn.data_name = cache_base_FCN;
else
    conf_fast_rcnn.data_name = share_data_FCN;
end
% conf_fast_rcnn.fcn_fg_thresh        = fcn_fg_thresh;
% conf_fast_rcnn.fcn_bg_thresh_hi     = fcn_bg_thresh_hi;
% conf_fast_rcnn.fcn_bg_thresh_lo     = fcn_bg_thresh_lo;
% conf_fast_rcnn.fcn_scales           = fcn_scales;
% conf_fast_rcnn.fcn_fg_fraction      = fcn_fg_fraction;
% conf_fast_rcnn.fcn_max_size         = fcn_max_size;
conf_fast_rcnn.update_roi_name      = update_roi_name;

% test data
dataset = [];
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);

%% multi-thres NMS setting
% factor_vec = [1 : -0.1 :0 ];
% %model.stage1_rpn.nms.scheme = 'no_minus';
% model.stage1_rpn.nms.scheme = 'minus';
% multi_NMS_setting = generate_multi_nms_setting();
% model.stage1_rpn.nms.nms_iou_thrs   = [0.90, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50];
% model.stage1_rpn.nms.max_per_image  = [3000, 2000, 1000, 800,  400,  200,  100,  50];

% for i = 1:length(factor_vec)
%     for j = 1:length(multi_NMS_setting)
%
%         model.stage1_rpn.nms.note = sprintf('multiNMS_fac_%.1f_set_%d_%s', ...
%             factor_vec(i), j, model.stage1_rpn.nms.scheme);   % must be a string
%
%         model.stage1_rpn.nms.nms_iou_thrs   = multi_NMS_setting(j).nms_iou_thrs;
%         model.stage1_rpn.nms.max_per_image  = multi_NMS_setting(j).max_per_image;
%         model.stage1_rpn.nms.factor = factor_vec(i);
%
%         %%  stage one proposal test
%         % test: compute recall and update roidb on TEST
%         cprintf('blue', '\nStage one proposal TEST on val data ...\n');
%         % dataset.roidb_test = RPN_TEST_ilsvrc_hyli(...
%         RPN_TEST_ilsvrc_hyli(...
%             'train14', 'final', model, ...
%             dataset.imdb_test, dataset.roidb_test, conf_proposal, ...
%             'mult_thr_nms',             model.stage1_rpn.nms.mult_thr_nms, ...
%             'nms_iou_thrs',             model.stage1_rpn.nms.nms_iou_thrs, ...
%             'max_per_image',            model.stage1_rpn.nms.max_per_image, ...
%             'update_roi',               update_roi, ...
%             'update_roi_name',          update_roi_name, ...
%             'skip_rpn_test',            skip_rpn_test, ...
%             'factor',                   model.stage1_rpn.nms.factor, ...
%             'scheme',                   model.stage1_rpn.nms.scheme, ...
%             'save_intermediate_box',    save_intermediate_box, ...
%             'gpu_id',                   opts.gpu_id ...
%             );
%     end;
% end;

%% stage two fast rcnn test
factor_vec = [1 : -0.1 :0 ];
fcn.nms.scheme = 'no_minus';
%fcn.nms.scheme = 'minus';
multi_NMS_setting = generate_multi_nms_setting_fcn();
test_sub_folder_suffix = 'no_minus_case';

for i = 1:length(factor_vec)
    for j = 1:length(multi_NMS_setting)
        
        cprintf('blue', '\nStage two Fast-RCNN cascade TEST...\n');
        
        fcn.nms.note = sprintf('multiNMS_fac_%.1f_set_%d_%s', ...
            factor_vec(i), j, fcn.nms.scheme);       
        fcn.nms.nms_iou_thrs   = multi_NMS_setting(j).nms_iou_thrs;
        fcn.nms.max_per_image  = multi_NMS_setting(j).max_per_image;
        fcn.nms.factor = factor_vec(i);
        
        fast_rcnn_test(conf_fast_rcnn, dataset.imdb_test, dataset.roidb_test, ...
            'net_def_file',             model.stage1_fast_rcnn.test_net_def_file, ...
            'net_file',                 fast_rcnn_net_file, ...
            'cache_name',               model.stage1_fast_rcnn.cache_name, ...
            'binary',                   binary_train, ...
            'max_per_image',            test_max_per_image, ...
            'avg_per_image',            test_avg_per_image, ...
            'nms',                      fcn.nms, ...
            'test_sub_folder_suffix',   test_sub_folder_suffix, ...
            'nms_overlap_thres',        fast_nms_overlap_thres, ...
            'after_nms_topN',           fast_rcnn_after_nms_topN ...
            );
    end
end