% RPN training and testing on ilsvrc
% 
% refactor by hyli on July 13 2016
%
% ---------------------------------------------------------

% clc; 
clear;
run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');

opts.caffe_version = 'caffe_faster_rcnn';
% whether or not do validation during training
opts.do_val = true;

% ======================= USER DEFINE =======================
% cache base
%cache_base_proposal = 'NEW_ilsvrc_vgg16_aaa';
cache_base_proposal = 'NEW_ILSVRC_ls139';
opts.gpu_id = 2;
% train14 only, plus val1
opts.train_key = 'train14';

% load paramters from the 'models' folder
%model = Model.VGG16_for_Faster_RCNN('solver_12w20w_ilsvrc');
model = Model.VGG16_for_Faster_RCNN('solver_10w30w_ilsvrc', 'test_original_anchor');
% finetune: uncomment the following if init from another model
%ft_file = './output/rpn_cachedir/NEW_ILSVRC_vgg16_stage1_rpn/train14/iter_75000.caffemodel';
model.anchor_size = 2.^(3:5);
model.ratios = [0.5, 1, 2];
detect_exist_config_file    = true;
detect_exist_train_file     = true;
use_flipped                 = true;     
update_roi                  = false;
% ==========================================================

model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, '', model);
% finetune
if exist('ft_file', 'var')
    net_file = ft_file;
    fprintf('\ninit from another model\n');
else
    net_file = model.stage1_rpn.init_net_file;
end
caffe_dir = './external/caffe/matlab';
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% config, must be input after setting caffe
% in the 'proposal_config.m' file
[conf_proposal, conf_fast_rcnn] = Faster_RCNN_Train.set_config( ...
    cache_base_proposal, model, detect_exist_config_file );

% train/test data
% init:
%   imdb_train, roidb_train, cell;
%   imdb_test, roidb_test, struct
dataset = [];
% change to point to your devkit install
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);
dataset = Dataset.ilsvrc14(dataset, opts.train_key, use_flipped, root_path);

%%  stage one proposal
fprintf('\nStage one proposal...\n');
% train
% model.stage1_rpn.output_model_file = proposal_train(...
%     conf_proposal, ...
%     dataset.imdb_train, dataset.roidb_train, opts.train_key, ...
%     'detect_exist_train_file',  detect_exist_train_file, ...
%     'do_val',                   opts.do_val, ...
%     'imdb_val',                 dataset.imdb_test, ...
%     'roidb_val',                dataset.roidb_test, ...
%     'solver_def_file',          model.stage1_rpn.solver_def_file, ...
%     'net_file',                 net_file, ...
%     'cache_name',               model.stage1_rpn.cache_name, ...
%     'snapshot_interval',        20000 ...
%     );
% fprintf('\nStage one DONE!\n');

% compute recall
% dataset = RPN_TEST_ilsvrc_hyli(cache_base_proposal, 'train14', 'iter_75000', ...
%     model, dataset, conf_proposal, 'update_roi', update_roi);

dataset = RPN_TEST_ilsvrc_hyli(cache_base_proposal, 'train14', 'final', ...
    model, dataset, conf_proposal, 'update_roi', update_roi);
