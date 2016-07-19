% Faster rcnn training and testing on ilsvrc
%
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------

function dataset = RPN_TEST_ilsvrc_hyli(cache_base_proposal, test_folder, ...
    iter_name, varargin)

% cache_base_proposal = 'NEW_ILSVRC_vgg16';
% cache_base_proposal = 'NEW_ilsvrc_vgg16_anchor_size';
% test_folder = 'ilsvrc14_val2';
% test_folder = 'train14';        % where the intermediate result resides
% iter_name = 'iter_20000';
% iter_name = 'final';
ip = inputParser;
ip.addRequired('cache_base_proposal',                       @isstr);
ip.addRequired('test_folder',                               @isstr);
ip.addRequired('iter_name',                                 @isstr);
ip.addRequired('model',                                     @isstruct);
ip.addRequired('dataset',                                   @isstruct);
ip.addRequired('conf_proposal',                             @isstruct);

ip.addParameter('gpu_id',               0,                  @isscalar);
ip.addParameter('update_roi',           false,              @islogical);
ip.parse(cache_base_proposal, test_folder, iter_name, varargin{:});
opts = ip.Results;

%% init
% opts.caffe_version = 'caffe_faster_rcnn';
% caffe_dir = './external/caffe/matlab';
% addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% load paramters from the 'models' folder
model = opts.model;
conf_proposal = opts.conf_proposal;
dataset = opts.dataset;

test_file = [test_folder '/'];
suffix = ['_' iter_name];

%% compute recall

cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', model.stage1_rpn.cache_name, dataset.imdb_test.name);
output_model_file = fullfile(pwd, 'output', 'rpn_cachedir', ...
    model.stage1_rpn.cache_name, test_file, [iter_name '.caffemodel']);

if exist(fullfile(cache_dir, ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']), 'file')
    
    fprintf('skip testing and directly load (%s) ...\n', ...
        ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']);
    
else
    % UPDATE: NO LONGER save 'aboxes' in the cache.
    % ==============
    % ==== TEST ====
    aboxes = proposal_test(conf_proposal, dataset.imdb_test, ...
        'net_def_file',     model.stage1_rpn.test_net_def_file, ...
        'net_file',         output_model_file, ...
        'cache_name',       model.stage1_rpn.cache_name, ...
        'suffix',           suffix);
    % ==============
    % ===== NMS ====
    % extremely time-consuming
    aboxes = boxes_filter_inline(aboxes, model.stage1_rpn.nms.per_nms_topN, ...
        model.stage1_rpn.nms.nms_overlap_thres, model.stage1_rpn.nms.after_nms_topN, conf_proposal.use_gpu);
    save(fullfile(cache_dir, ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']), 'aboxes', '-v7.3');
end

recall_per_cls = compute_recall_ilsvrc(...
    fullfile(cache_dir, ['aboxes_filtered_' dataset.imdb_test.name suffix '.mat']), 300);

mean_recall = mean(extractfield(recall_per_cls, 'recall'));
save(fullfile(cache_dir, ['recall_' dataset.imdb_test.name suffix ...
    sprintf('_%.2f.mat', 100*mean_recall)]), 'recall_per_cls');

if opts.update_roi
    
    roidb_regions.boxes = aboxes;
    roidb_regions.images = dataset.imdb_test.image_ids;
    
    % update: change some code to save memory
    %     try
    %         ld = load(fullfile(cache_dir, 'trick_new_roidb.mat'));
    %         rois = ld.rois;
    %         assert(length(rois) == length(roidb.rois));
    %         clear ld;
    %     catch
    fprintf('update roidb.rois during test, taking quite a while (brew some coffe or take a walk!:)...\n');
    
    % save the file 'trick_new_roidb.mat'
    roidb_from_proposal(dataset.imdb_test, dataset.roidb_test, ...
        roidb_regions, 'keep_raw_proposal', false, 'mat_file_prefix', cache_dir);
    
    ld = load(fullfile(cache_dir, 'trick_new_roidb.mat'));
    rois = ld.rois;
    assert(length(rois) == length(roidb.rois));
    clear ld;
    %     end
    dataset.roidb_test.rois = rois;
end
