% Faster rcnn training and testing on ilsvrc
%
% refactor by hyli on May 15 2016
% note:
%       just some stupid task assigned by damn Wanli Ouyang
% ---------------------------------------------------------

function roidb = RPN_TEST_ilsvrc_hyli(...
    cache_base_proposal, test_folder, iter_name, ...
    varargin)

ip = inputParser;
ip.addRequired('cache_base_proposal',                       @isstr);
% 'test foler' means where the trained model (.caffemodel) resides.
ip.addRequired('test_folder',                               @isstr);
ip.addRequired('iter_name',                                 @isstr);
ip.addRequired('model',                                     @isstruct);
ip.addRequired('imdb',                                      @isstruct);
ip.addRequired('roidb',                                      @isstruct);
ip.addRequired('conf_proposal',                             @isstruct);

% by default all test programs use GPU=0
ip.addParameter('mult_thr_nms',         false,              @islogical);
ip.addParameter('nms_iou_thrs',         [0.95, 0.90, 0.85, 0.80, 0.75, 0.65, 0.60, 0.55],  @isnumeric);
ip.addParameter('max_per_image',        [2000, 1000,  400,  200,  100,   40,   20,   10],  @isnumeric);
ip.addParameter('gpu_id',               0,                  @isscalar);
ip.addParameter('update_roi',           false,              @islogical);
ip.addParameter('update_roi_name',      '',                 @isstr);
ip.parse(cache_base_proposal, test_folder, iter_name, varargin{:});
opts = ip.Results;

%% init
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% load paramters from the 'models' folder
model = opts.model;
conf_proposal = opts.conf_proposal;
imdb = opts.imdb;
roidb = opts.roidb;

test_file = [test_folder '/'];
suffix = ['_' iter_name];

%% compute recall
if isnan(str2double(model.stage1_rpn.nms.note))
    % multi-thres NMS
    detect_name = {model.stage1_rpn.nms.note};
else
    detect_name = arrayfun(@(x) num2str(x), ...
        model.stage1_rpn.nms.nms_overlap_thres, 'uniformoutput', false);
end

clear raw_aboxes;
forward_flag = false;

for i = 1:length(detect_name)
    
    clear aboxes;
    cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', ...
        model.stage1_rpn.cache_name, imdb.name);
    test_box_full_name = fullfile(cache_dir, ...
        ['aboxes_filtered_' imdb.name suffix ...
        sprintf('_NMS_%s.mat', detect_name{i})]);
    
    % 1. get the 'aboxes_filtered_xx' files
    if exist(test_box_full_name, 'file')
        
        fprintf('skip testing and directly load (%s) ...\n', ...
            ['aboxes_filtered_' imdb.name suffix ...
            sprintf('_NMS_%s.mat', detect_name{i})]);
    else
        
        if ~forward_flag
            
            % ==============
            % ==== TEST ====
            % UPDATE: NO LONGER save 'aboxes' in the cache.
            disp('nms:');
            disp(model.stage1_rpn.nms);
            
            output_model_file = fullfile(pwd, 'output', 'rpn_cachedir', ...
                model.stage1_rpn.cache_name, test_file, [iter_name '.caffemodel']);
            
            raw_aboxes = proposal_test(conf_proposal, ...
                imdb, ...
                'net_def_file',     model.stage1_rpn.test_net_def_file, ...
                'net_file',         output_model_file, ...
                'cache_name',       model.stage1_rpn.cache_name, ...
                'suffix',           suffix);
            % only execute once
            forward_flag = true;
        end
        
        % ==============
        % ===== NMS ====
        % extremely time-consuming
        if ~opts.mult_thr_nms
            aboxes = boxes_filter_inline(raw_aboxes, ...
                model.stage1_rpn.nms.per_nms_topN, ...          % -1
                model.stage1_rpn.nms.nms_overlap_thres(i), ...     % 0.6, 0.7, ...
                model.stage1_rpn.nms.after_nms_topN, ...        % 2000
                conf_proposal.use_gpu);
        else
            fprintf('do multi-thres nms, taking quite a while (brew some coffe or take a walk!:)...\n');
            parfor kk = 1:length(raw_aboxes)
                aboxes{kk} = AttractioNet_postprocess(raw_aboxes{kk}, 'thresholds', -inf, 'use_gpu', true, ...
                    'mult_thr_nms',     true, ...
                    'nms_iou_thrs',     opts.nms_iou_thrs, ...
                    'max_per_image',    opts.max_per_image);
            end
        end
        save(test_box_full_name, 'aboxes', '-v7.3');
    end
    
    % 2. compute recall
    recall_per_cls = compute_recall_ilsvrc(test_box_full_name, 300);
    mean_recall = mean(extractfield(recall_per_cls, 'recall'));
    fprintf('model:: %s, (nms) %s, mean rec:: %.2f\n\n', iter_name, detect_name{i}, 100*mean_recall);
    
    % 3. save the detailed recall file
    save(fullfile(cache_dir, ['recall_' imdb.name suffix ...
        sprintf('_%.2f_NMS_%s.mat', 100*mean_recall, detect_name{i})]), ...
        'recall_per_cls');
end

if opts.update_roi
    
    assert(length(model.stage1_rpn.nms.nms_overlap_thres) == 1);
    
    if imdb.flip, FLIP = 'flip'; else FLIP = 'unflip'; end  
    update_roi_file = fullfile(pwd, 'imdb/cache/ilsvrc', ...
        ['roidb_' roidb.name ...
        '_' FLIP sprintf('_%s.mat', opts.update_roi_name)]);
      
    if ~exist(update_roi_file, 'file')
        
        fprintf('update roidb.rois, taking quite a while ...\n');
        if ~exist('aboxes', 'var'), load(test_box_full_name); end
        roidb_regions = [];
        roidb_regions.boxes = aboxes;
        roidb_regions.images = imdb.image_ids;
        % update roidb in 'imdb' folder
        roidb_from_proposal(imdb, roidb, ...
            roidb_regions, 'keep_raw_proposal', false, 'mat_file', update_roi_file);
    end
    
    ld = load(update_roi_file);
    rois_load = ld.rois;
    assert(length(rois_load) == length(roidb.rois));
    clear ld;
    % update in matlab dynamic memory
    roidb.rois = rois_load;
end
