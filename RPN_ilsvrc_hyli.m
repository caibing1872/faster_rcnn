% Faster rcnn training and testing on ilsvrc
% 
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------

% update on May 23, 2016
%   
%   1. skip data prep if training data (201, train+val1) is ready
%   2. save loss info dynamically 
%   3. resume
%   4. test recall instead of stupid test loss


clc; clear;
run('./startup');
%% init
fprintf('\nInitialize model, dataset, and configuration...\n');
opts.caffe_version = 'caffe_faster_rcnn';
opts.gpu_id = 0;
opts.do_val = true;                 % whether or not do testing(val) during training
opts.skip_data_prep = false;        % if training data is READY (valid when key is 'train')
%opts.train_key = 'train_val1';      % 'train', 'train_val1'
opts.train_key = 'train'; 

caffe_dir = './external/caffe/matlab';
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% load paramters from the 'models' folder
model = Model.VGG16_for_Faster_RCNN;
% cache base
cache_base_proposal = 'ilsvrc_vgg16_try';
cache_base_fast_rcnn = '';
model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, ...
    cache_base_fast_rcnn, model);

% train/test data
% init:
%   imdb_train, roidb_train, cell;
%   imdb_test, roidb_test, struct
dataset = [];
if ~opts.skip_data_prep || strcmp(opts.train_key, 'train_val1')
    % change to point to your devkit install
    root_path = './datasets/ilsvrc14_det';
    %root_path = '/home/hongyang/dataset/imagenet_det';
    use_flipped = true;
    %dataset = Dataset.ilsvrc14(dataset, 'train', use_flipped, root_path);
    dataset = Dataset.ilsvrc14(dataset, opts.train_key, use_flipped, root_path);
    dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);
else
    dataset.imdb_train = cell(1);
    dataset.imdb_test.name = 'ilsvrc14_val2';
    dataset.roidb_train = cell(1);
    dataset.roidb_test = struct();
end

% config
[ conf_proposal, conf_fast_rcnn ] = ...
    Faster_RCNN_Train.set_config( cache_base_proposal, model );

%%  stage one proposal
fprintf('\nStage one proposal...\n');
% train
model.stage1_rpn.output_model_file = proposal_train_chunk(...
    conf_proposal, ...
    dataset.imdb_train, dataset.roidb_train, ...                                  
    'do_val',           opts.do_val, ...
    'imdb_val',         dataset.imdb_test, ...
    'roidb_val',        dataset.roidb_test, ...
    'solver_def_file',  model.stage1_rpn.solver_def_file, ...
    'net_file',         model.stage1_rpn.init_net_file, ...
    'cache_name',       model.stage1_rpn.cache_name ...
    );

% % test
% dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
%     model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);



