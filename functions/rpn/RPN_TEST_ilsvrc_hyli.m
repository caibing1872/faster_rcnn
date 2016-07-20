% Faster rcnn training and testing on ilsvrc
%
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------

function dataset = RPN_TEST_ilsvrc_hyli(cache_base_proposal, test_folder, ...
    iter_name, varargin)

ip = inputParser;
ip.addRequired('cache_base_proposal',                       @isstr);
% 'test foler' means where the trained model (.caffemodel) resides.
ip.addRequired('test_folder',                               @isstr);
ip.addRequired('iter_name',                                 @isstr);
ip.addRequired('model',                                     @isstruct);
ip.addRequired('dataset',                                   @isstruct);
ip.addRequired('conf_proposal',                             @isstruct);

% by default all test programs use GPU=0
ip.addParameter('mult_thr_nms',         false,              @islogical);
ip.addParameter('nms_iou_thrs',         [0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.60, 0.55],  @isnumeric);
ip.addParameter('max_per_image',        [2000, 1000,  400,  200,  100,   40,   20,   10],  @isnumeric);
ip.addParameter('gpu_id',               0,                  @isscalar);
ip.addParameter('update_roi',           false,              @islogical);
ip.parse(cache_base_proposal, test_folder, iter_name, varargin{:});
opts = ip.Results;

%% init
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
cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', ...
    model.stage1_rpn.cache_name, dataset.imdb_test.name);
test_box_full_name = fullfile(cache_dir, ...
    ['aboxes_filtered_' dataset.imdb_test.name suffix ...
    sprintf('_NMS_%s.mat', model.stage1_rpn.nms.note)]);

output_model_file = fullfile(pwd, 'output', 'rpn_cachedir', ...
    model.stage1_rpn.cache_name, test_file, [iter_name '.caffemodel']);
if exist(test_box_full_name, 'file')
    
    fprintf('skip testing and directly load (%s) ...\n', ...
        ['aboxes_filtered_' dataset.imdb_test.name suffix ...
        sprintf('_NMS_%s.mat', model.stage1_rpn.nms.note)]);
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
    
    if ~opts.mult_thr_nms
        aboxes = boxes_filter_inline(aboxes, model.stage1_rpn.nms.per_nms_topN, ...
            model.stage1_rpn.nms.nms_overlap_thres, model.stage1_rpn.nms.after_nms_topN, conf_proposal.use_gpu);
    else
        aboxes = AttractioNet_postprocess(aboxes, 'thresholds', -inf, 'use_gpu', true, ...
            'mult_thr_nms',     true, ...
            'nms_iou_thrs',     opts.nms_iou_thrs, ...
            'max_per_image',    opts.max_per_image);
    end
    
    save(test_box_full_name, 'aboxes', '-v7.3');
end

recall_per_cls = compute_recall_ilsvrc(test_box_full_name, 300);
mean_recall = mean(extractfield(recall_per_cls, 'recall'));
fprintf('model:: %s, mean rec:: %.2f\n\n', iter_name, 100*mean_recall);
save(fullfile(cache_dir, ['recall_' dataset.imdb_test.name suffix ...
    sprintf('_%.2f_NMS_%s.mat', 100*mean_recall, model.stage1_rpn.nms.note)]), ...
    'recall_per_cls');

if opts.update_roi
    
    roidb_regions = [];
    roidb_regions.boxes = aboxes;
    roidb_regions.images = dataset.imdb_test.image_ids;
    
    if dataset.imdb_test.flip, FLIP = 'flip'; else FLIP = 'unflip'; end
    PREDEFINED = fullfile(pwd, 'imdb/cache/ilsvrc');
    % update: change some code to save memory
    %     try
    %         ld = load(fullfile(cache_dir, 'trick_new_roidb.mat'));
    %         rois = ld.rois;
    %         assert(length(rois) == length(roidb.rois));
    %         clear ld;
    %     catch
    fprintf('update roidb.rois during test, taking quite a while ...\n');
    
    % update in 'imdb' folder
    roidb_from_proposal(dataset.imdb_test, dataset.roidb_test, ...
        roidb_regions, 'keep_raw_proposal', false, 'mat_file_prefix', PREDEFINED);
    
    ld = load(fullfile(PREDEFINED, ['roidb_' roidb.name '_' FLIP '_1.mat']));
    rois = ld.rois;
    assert(length(rois) == length(dataset.roidb_test.rois));
    clear ld;
    %     end
    % update in matlab dynamic memory
    dataset.roidb_test.rois = rois;
end
