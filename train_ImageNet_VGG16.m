% script_faster_rcnn_ImageNet_VGG16()
% Faster rcnn training and testing with VGG16 model
% --------------------------------------------------------

clc; clear;
run('./startup');
%% init
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 1;
opts.do_val                 = true;

caffe_dir = './external/caffe/matlab';
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% load paramters from the 'models' folder (same as VOC07)
model = Model.VGG16_for_Faster_RCNN_VOC2007;

% train/test data
dataset                     = [];
use_flipped                 = true;
% change to point to your devkit install
devkit = './datasets/ilsvrc13_det';
dataset = Dataset.ilsvrc13(dataset, 'train', use_flipped, devkit);
dataset = Dataset.ilsvrc13(dataset, 'test', false, devkit);

% conf
conf_proposal = proposal_config('image_means', model.mean_image, ...
    'feat_stride', model.feat_stride);
conf_fast_rcnn = fast_rcnn_config('image_means', model.mean_image);

% cache base
cache_base_proposal         = 'ilsvrc13_vgg';
cache_base_fast_rcnn        = '';
% set cache folder for each stage
model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);

% generate anchors and pre-calculate output size of rpn network 
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
    = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn = Faster_RCNN_Train.do_proposal_train(conf_proposal, ...
    dataset, model.stage1_rpn, opts.do_val);

% test
dataset.roidb_train = cellfun(@(x, y) ...
    Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y), ...
    dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);

dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
    model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage one fast rcnn
fprintf('\n***************\nstage one fast rcnn\n***************\n');
% train
model.stage1_fast_rcnn = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, ...
    dataset, model.stage1_fast_rcnn, opts.do_val);
% test
opts.mAP = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, ...
    model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

%%  stage two proposal
% net proposal
fprintf('\n***************\nstage two proposal\n***************\n');
% train
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn = Faster_RCNN_Train.do_proposal_train(conf_proposal, ...
    dataset, model.stage2_rpn, opts.do_val);
% test
dataset.roidb_train = cellfun(@(x, y) ...
    Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, x, y), ...
    dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);

dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
    model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

%%  stage two fast rcnn
fprintf('\n***************\nstage two fast rcnn\n***************\n');
% train
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, ...
    dataset, model.stage2_fast_rcnn, opts.do_val);

%%  final test
fprintf('\n***************\nfinal test\n***************\n');
     
model.stage2_rpn.nms = model.final_test.nms;

dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
    model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

opts.final_mAP = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, ...
    model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% save final models, for outside tester
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, ...
    conf_fast_rcnn, model, dataset);