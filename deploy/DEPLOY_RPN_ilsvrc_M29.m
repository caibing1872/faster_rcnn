% RPN training and testing on ilsvrc
% 
% refactor by hyli on July 13 2016
% This file should descend from 'RPN_ilsvrc_hyli_new.m' and always be updated with master file.
% ---------------------------------------------------------

clc; clear;
run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');

opts.caffe_version = 'caffe_faster_rcnn';
% whether or not do validation during training
opts.do_val = true;

% ======================= USER DEFINE =======================
% cache base
cache_base_proposal = 'M29_s31';
opts.gpu_id = 0;
% train14 only, plus val1
opts.train_key = 'train14';

% load paramters from the 'models' folder
model = Model.VGG16_for_Faster_RCNN('solver_15w45w_ilsvrc_25anchor', ...
    'test_25anchor');
% finetune: uncomment the following if init from another model
% ft_file = './output/rpn_cachedir/NEW_ILSVRC_vgg16_stage1_rpn/train14/iter_75000.caffemodel';

detect_exist_config_file    = true;
detect_exist_train_file     = true;
use_flipped                 = true;     
update_roi                  = false;
model.anchor_size = 2.^(4:8);       % 25 anchors
model.ratios = [0.333, 0.5, 1, 2, 3];
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

conf_proposal.cache_base_proposal = cache_base_proposal;
% ================= following experiments on s31 ===========
conf_proposal.fg_thresh = 0.7;
conf_proposal.bg_thresh_hi = 0.3;
conf_proposal.scales = [800];
conf_proposal.test_scales = [800];
% ==========================================================

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
model.stage1_rpn.output_model_file = proposal_train(...
    conf_proposal, ...
    dataset.imdb_train, dataset.roidb_train, opts.train_key, ...
    'detect_exist_train_file',  detect_exist_train_file, ...
    'do_val',                   opts.do_val, ...
    'imdb_val',                 dataset.imdb_test, ...
    'roidb_val',                dataset.roidb_test, ...
    'solver_def_file',          model.stage1_rpn.solver_def_file, ...
    'net_file',                 net_file, ...
    'cache_name',               model.stage1_rpn.cache_name, ...
    'snapshot_interval',        20000 ...
    );
fprintf('\nStage one DONE!\n');

% compute recall and update roidb on TEST
dataset = RPN_TEST_ilsvrc_hyli(cache_base_proposal, 'train14', 'iter_75000', ...
    model, dataset, conf_proposal, 'update_roi', update_roi);

%% fast rcnn train
% model_stage.output_model_file = fast_rcnn_train(conf, dataset.imdb_train, dataset.roidb_train, ...
%                                 'do_val',           do_val, ...
%                                 'imdb_val',         dataset.imdb_test, ...
%                                 'roidb_val',        dataset.roidb_test, ...
%                                 'solver_def_file',  model_stage.solver_def_file, ...
%                                 'net_file',         model_stage.init_net_file, ...
%                                 'cache_name',       model_stage.cache_name);
