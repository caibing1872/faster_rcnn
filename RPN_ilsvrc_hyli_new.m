% Faster rcnn training and testing on ilsvrc
% 
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------
%
% update on May 29, 2016
%   
%   1. skip data prep if training data is ready
%   2. save loss info dynamically 
%   3. resume
%   4. test recall instead of stupid test loss
clc; clear;
run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');
opts.caffe_version = 'caffe_faster_rcnn';
% whether or not do testing(val) during training
opts.do_val = true;

% =====================================================
% cache base
cache_base_proposal = 'NEW_ilsvrc_vgg16_conti';
%cache_base_proposal = 'NEW_ILSVRC_vgg16_ls139';
opts.gpu_id = 0;
opts.train_key = 'train14';                     % train14 only, plus val1
% load paramters from the 'models' folder
%model = Model.VGG16_for_Faster_RCNN('solver_12w20w_ilsvrc');
model = Model.VGG16_for_Faster_RCNN('solver_60k80k_conti');
% uncomment the following if init from another model
ft_file = './output/rpn_cachedir/NEW_ILSVRC_vgg16_stage1_rpn/train14/iter_75000.caffemodel';
%model = Model.VGG16_for_Faster_RCNN('solver_8w13w');
use_flipped = false;     % ls139 has flip version
% =====================================================

model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, '', model);
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
[conf_proposal, ~] = Faster_RCNN_Train.set_config( cache_base_proposal, model );

% train/test data
% init:
%   imdb_train, roidb_train, cell;
%   imdb_test, roidb_test, struct
dataset = [];
% change to point to your devkit install
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, opts.train_key, use_flipped, root_path);
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);

%%  stage one proposal
fprintf('\nStage one proposal...\n');
% train
model.stage1_rpn.output_model_file = proposal_train(...
    conf_proposal, ...
    dataset.imdb_train, dataset.roidb_train, opts.train_key, ...
    'do_val',               opts.do_val, ...
    'imdb_val',             dataset.imdb_test, ...
    'roidb_val',            dataset.roidb_test, ...
    'solver_def_file',      model.stage1_rpn.solver_def_file, ...
    'net_file',             net_file, ...
    'cache_name',           model.stage1_rpn.cache_name, ...
    'snapshot_interval',    10000, ...
    'solverstate',          '' ...
    );

% final test
dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
    model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
