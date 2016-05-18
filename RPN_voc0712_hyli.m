% Faster rcnn training and testing
% 
% refactor by hyli on Apr 30 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------

clc; clear;
run('./startup');
%% init
fprintf('\n\nInitialize model and dataset configurations...\n');
opts.caffe_version = 'caffe_faster_rcnn';
opts.gpu_id = 0;
opts.do_val = true;

caffe_dir = './external/caffe/matlab';
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

opts.debug = false;
% === MODEL:: load paramters from the 'models' folder
% it's the same among datasets
%model = Model.VGG19_voc07;        
%cache_base_proposal = 'VOC0712_vgg19';
cache_base_proposal = 'VOC0712_vgg16_NEW_139';
% cache_base_proposal = 'VOC07_vgg19';

model = Model.VGG16_for_Faster_RCNN_VOC2007;
% cache_base_proposal = 'VOC07_vgg';

cache_base_fast_rcnn = '';
model = Faster_RCNN_Train.set_cache_folder(...
    cache_base_proposal, cache_base_fast_rcnn, model);

% config
[ conf_proposal, conf_fast_rcnn ] = ...
    Faster_RCNN_Train.set_config( cache_base_proposal, model );

% === DATA:: train/test data
% init:
%   imdb_train, roidb_train, cell;
%   imdb_test, roidb_test, struct
dataset = [];
use_flipped = true;
dataset = Dataset.voc0712_trainval(dataset, 'train', use_flipped);
%dataset = Dataset.voc2007_trainval(dataset, 'train', use_flipped);
dataset = Dataset.voc2007_test(dataset, 'test', false);

%%  stage one proposal
fprintf('\n\nStage one proposal...\n');
% % train
% model.stage1_rpn.output_model_file = proposal_train(...
%     conf_proposal, dataset.imdb_train, dataset.roidb_train, ...                                  
%     'do_val',           opts.do_val, ...
%     'imdb_val',         dataset.imdb_test, ...
%     'roidb_val',        dataset.roidb_test, ...
%     'solver_def_file',  model.stage1_rpn.solver_def_file, ...
%     'net_file',         model.stage1_rpn.init_net_file, ...
%     'cache_name',       model.stage1_rpn.cache_name, ...
%     'debug',            opts.debug);

% test
% dataset.roidb_train = cellfun(@(x, y) ...
%     Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y), ...
%     dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);

dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
    model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);


