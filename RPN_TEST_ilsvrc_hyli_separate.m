% Faster rcnn training and testing on ilsvrc
% 
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------

function RPN_TEST_ilsvrc_hyli_separate(cache_base_proposal, test_folder, iter_name)

% cache_base_proposal = 'NEW_ILSVRC_vgg16';
cache_base_proposal = 'NEW_ilsvrc_vgg16_anchor_size';
%test_folder = 'ilsvrc14_val2';
test_folder = 'train14';        % where the intermediate result resides
iter_name = 'iter_20000';
%iter_name = 'final';
%% init
opts.caffe_version = 'caffe_faster_rcnn';
opts.gpu_id = 1;

caffe_dir = './external/caffe/matlab';
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% load paramters from the 'models' folder
model = Model.VGG16_for_Faster_RCNN;
test_file = [test_folder '/'];
suffix = ['_' iter_name];

model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, '', model);
% config
[ conf_proposal, ~ ] =  Faster_RCNN_Train.set_config( cache_base_proposal, model );
% test data
dataset = [];
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);

%%  stage one test and compute recall
% test
% dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, ...
%     model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);

% revised by hyli
cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', model.stage1_rpn.cache_name, dataset.imdb_test.name);
output_model_file = fullfile(pwd, 'output', 'rpn_cachedir', ...
    model.stage1_rpn.cache_name, test_file, [iter_name '.caffemodel']);
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
mean_recall = mean(extractfield(recall_per_cls, 'recall'));
save(fullfile(cache_dir, ['recall_' dataset.imdb_test.name suffix ...
    sprintf('_%.2f.mat', 100*mean_recall)]), 'recall_per_cls');
