% Faster rcnn training and testing on ilsvrc
% 
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------


%% init
opts.caffe_version = 'caffe_faster_rcnn';
opts.gpu_id = 0;

caffe_dir = './external/caffe/matlab';
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% load paramters from the 'models' folder
model = Model.VGG16_for_Faster_RCNN;

% cache_base_proposal = 'ilsvrc_vgg16_val1';
% test_file = 'ilsvrc14_val1/final';
% suffix = '_final';
%
cache_base_proposal = 'ilsvrc_vgg16_train14';
test_file = 'ilsvrc14_train14/iter_90000';
suffix = '_iter_90000';

cache_base_fast_rcnn = '';
model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, ...
    cache_base_fast_rcnn, model);

% config
[ conf_proposal, ~ ] =  Faster_RCNN_Train.set_config( cache_base_proposal, model );

% train/test data
% init:
%   imdb_train, roidb_train, cell;
%   imdb_test, roidb_test, struct
dataset = [];
% change to point to your devkit install
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);


%%  stage one test and compute recall
% test
% dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
%     model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

% revised by hyli
cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', model.stage1_rpn.cache_name, dataset.imdb_test.name);
output_model_file = fullfile(pwd, 'output', 'rpn_cachedir', model.stage1_rpn.cache_name, test_file);
try
    ld = load(fullfile(cache_dir, ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']));
    aboxes = ld.aboxes;
    fprintf('skip testing...\n');
    clear ld;
catch
    % save 'aboxes' in the cache:
    %   proposal_boxes_ilsvrc13_val1.mat
    % ==============
    % ==== TEST ====
    aboxes = proposal_test(conf_proposal, dataset.imdb_test, ...
        'net_def_file',     model.stage1_rpn.test_net_def_file, ...
        'net_file',         output_model_file, ...
        'cache_name',       model.stage1_rpn.cache_name, ...
        'suffix',           suffix);
    
    % NMS, the following is extremely time-consuming
    aboxes = boxes_filter_inline(aboxes, model.stage1_rpn.nms.per_nms_topN, ...
        model.stage1_rpn.nms.nms_overlap_thres, model.stage1_rpn.nms.after_nms_topN, conf_proposal.use_gpu);
    % aboxes: test_num x 1 cell, each entry: 
    save(fullfile(cache_dir, ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']), 'aboxes', '-v7.3');
end

recall_per_cls = compute_recall_ilsvrc(...
    fullfile(cache_dir, ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']), 300);
save(fullfile(cache_dir, ['recall_' dataset.imdb_test.name suffix '.mat']), 'recall_per_cls');
