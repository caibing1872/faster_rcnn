function mAP = fast_rcnn_test(conf, imdb, roidb, varargin)
% mAP = fast_rcnn_test(conf, imdb, roidb, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
mAP = 0;
caffe.reset_all();
%% inputs
ip = inputParser;
ip.addRequired('conf',                              @isstruct);
ip.addRequired('imdb',                              @isstruct);
ip.addRequired('roidb',                             @isstruct);
ip.addParameter('net_def_file',     '',             @isstr);
ip.addParameter('net_file',         '',             @iscell);
ip.addParameter('cache_name',       '',             @isstr);
ip.addParameter('suffix',           '',             @isstr);
ip.addParameter('ignore_cache',     false,          @islogical);
ip.addParameter('binary',           true,           @islogical);
ip.addParameter('max_per_image',    100,            @isscalar);
ip.addParameter('avg_per_image',    40,             @isscalar);
ip.addParameter('bulk_prefix',      '',             @isstr);
ip.addParameter('test_sub_folder_suffix', '',       @isstr);
% normal nms
ip.addParameter('nms_overlap_thres',0,              @ismatrix);
ip.addParameter('after_nms_topN',   2000,           @isscalar);
% multi-thres nms
ip.addParameter('nms',              '',             @isstruct);
ip.parse(conf, imdb, roidb, varargin{:});
opts = ip.Results;
per_nms_topN = -1;
nms_overlap_thres = opts.nms_overlap_thres;
after_nms_topN = opts.after_nms_topN;

%%  set cache dir
cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', opts.cache_name, imdb.name);
if ~isempty(opts.bulk_prefix)
    cache_dir = fullfile(opts.bulk_prefix, imdb.name);
end
mkdir_if_missing(cache_dir);
if isempty(opts.test_sub_folder_suffix), cache_dir_sub = fullfile(cache_dir, opts.net_file{2});
else cache_dir_sub = fullfile(cache_dir, [opts.net_file{2} '_' opts.test_sub_folder_suffix]); end
mkdir_if_missing(cache_dir_sub);

%%  init log
% init matlab log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'matlab_log'));
log_file = fullfile(cache_dir, 'matlab_log', ['test_', timestamp, '.txt']);
diary(log_file);
% init caffe log
mkdir_if_missing([cache_dir '/caffe_log']);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log/test_');
caffe.init_log(caffe_log_file_base);
% init caffe net
caffe_net_file_full = fullfile(pwd, 'output', 'fast_rcnn_cachedir', ...
    opts.cache_name, opts.net_file{1}, sprintf('%s.caffemodel', opts.net_file{2}));

if ~isempty(opts.bulk_prefix)
    caffe_net_file_full = fullfile(opts.bulk_prefix, ...
        opts.net_file{1}, sprintf('%s.caffemodel', opts.net_file{2}));
end

num_images = length(imdb.image_ids);
num_classes = imdb.num_classes;
if opts.binary, num_classes = 1; imdb.classes{1} = 'binary'; end
% heuristic: keep an average of 40 detections per class per image prior to NMS
max_per_set = opts.avg_per_image * num_images;
% heuristic: keep at most 100 detection per class per image prior to NMS
max_per_image = opts.max_per_image;


% save the 'binary_xx' raw boxes before nms
save_file = @(x) fullfile(cache_dir_sub, ...
    [imdb.classes{x} '_boxes_' ...
    imdb.name opts.suffix ...
    sprintf('_max_%d_avg_%d.mat', max_per_image, opts.avg_per_image)]);

% save the boxes after nms
save_after_nms = @(x, y, z) fullfile(cache_dir_sub, ...
    [imdb.classes{x} '_boxes_' ...
    imdb.name opts.suffix sprintf('_nms_%.2f_topN_%d.mat', y, z)]);
%% testing
show_num = 3000;
% skip_fast_rcnn_test
try
    aboxes = cell(num_classes, 1);
    if opts.ignore_cache
        throw('');
    end
    for i = 1:num_classes
        load(save_file(i));
        aboxes{i} = boxes;
    end
    disp('boxes exist. skip testing ...');
catch
    disp('opts:');
    disp(opts);
    disp('conf:');
    disp(conf);
    
    caffe_net = caffe.Net(opts.net_def_file, 'test');
    caffe_net.copy_from(caffe_net_file_full);
    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);
    
    % determine the maximum number of rois in testing
    max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);
    % double check first
    [~, scores_check] = fast_rcnn_im_detect(conf, caffe_net, ...
        imread(imdb.image_at(1)), roidb.rois(1).boxes, max_rois_num_in_gpu);
    assert(size(scores_check, 1) == size(roidb.rois(1).gt,1));
    assert(size(scores_check, 2) == num_classes);
    
    % detection thresold for each class (this is adaptively set based on the max_per_set constraint)
    thresh = -inf * ones(num_classes, 1);
    % top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
    top_scores = cell(num_classes, 1);
    % all detections are collected into:
    %    all_boxes[cls][image] = N x 5 array of detections in
    %    (x1, y1, x2, y2, score)
    aboxes = cell(num_classes, 1);
    box_inds = cell(num_classes, 1);
    for i = 1:num_classes
        aboxes{i} = cell(length(imdb.image_ids), 1);
        box_inds{i} = cell(length(imdb.image_ids), 1);
    end
    count = 0;
    t_start = tic;
    
    % step 1
    for i = 1:num_images
        count = count + 1;
        %fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
        if i == 1 || i == num_images || mod(i, show_num)==0
            fprintf('%s: test %s (%s) %d/%d \n', procid(), ...
                opts.suffix, imdb.name, count, num_images);
        end
        %th = tic;
        d = roidb.rois(i);
        im = imread(imdb.image_at(i));
        [boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im, d.boxes, max_rois_num_in_gpu);
        
        for j = 1:num_classes
            inds = find(~d.gt & scores(:, j) > thresh(j));
            if ~isempty(inds)
                [~, ord] = sort(scores(inds, j), 'descend');
                ord = ord(1:min(length(ord), max_per_image));
                inds = inds(ord);
                
                cls_boxes = boxes(inds, (1+(j-1)*4):((j)*4));
                cls_scores = scores(inds, j);
                aboxes{j}{i} = [aboxes{j}{i}; cat(2, single(cls_boxes), single(cls_scores))];
                box_inds{j}{i} = [box_inds{j}{i}; inds];
            else
                aboxes{j}{i} = [aboxes{j}{i}; zeros(0, 5, 'single')];
                box_inds{j}{i} = box_inds{j}{i};
            end
        end
        
        % fprintf(' time: %.3fs\n', toc(th));
        % update 'thres'
        if mod(count, 1000) == 0 || count == num_images
            for j = 1:num_classes
                [aboxes{j}, box_inds{j}, thresh(j)] = ...
                    keep_top_k(aboxes{j}, box_inds{j}, i, max_per_set, thresh(j));
            end
            fprintf(' || thres at %d is %.4f\n', count, thresh);
        end
    end
    
    % step 2
    % 'top_scores' is fucking unused
    for i = 1:num_classes
        top_scores{i} = sort(top_scores{i}, 'descend');
        if (length(top_scores{i}) > max_per_set)
            thresh(i) = top_scores{i}(max_per_set);
        end
        % go back through and prune out detections below the found threshold
        for j = 1:length(imdb.image_ids)
            if ~isempty(aboxes{i}{j})
                I = find(aboxes{i}{j}(:,end) < thresh(i));
                aboxes{i}{j}(I,:) = [];
                box_inds{i}{j}(I,:) = [];
            end
        end
        boxes = aboxes{i};
        inds = box_inds{i};
        save(save_file(i), 'boxes', 'inds');
        clear boxes inds;
    end
    fprintf('test all images in %f seconds.\n', toc(t_start));
    
    caffe.reset_all();
    rng(prev_rng);
end

compute_recall_switch = true;
if strcmp(imdb.name, 'ilsvrc14_val2_no_GT') || ...
        strcmp(imdb.name, 'ilsvrc14_val1_13') || ...
        strcmp(imdb.name, 'ilsvrc14_val1_14') || ...
        strcmp(imdb.name, 'ilsvrc14_real_test') || ...
        strcmp(imdb.name, 'ilsvrc14_pos1k_13')
    compute_recall_switch = false;
end

if ~opts.binary
    % ------------------------------------------------------------------------
    % Peform AP evaluation (do nms inside the eval code)
    % ------------------------------------------------------------------------
    if isequal(imdb.eval_func, @imdb_eval_voc)
        % pascal voc
        for model_ind = 1:num_classes
            cls = imdb.classes{model_ind};
            res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, opts.cache_name, opts.suffix);
        end
    else
        % ilsvrc
        res = imdb.eval_func(aboxes, imdb, opts.cache_name, opts.suffix);
    end
    if ~isempty(res)
        fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
        fprintf('Results:\n');
        aps = [res(:).ap]' * 100;
        disp(aps);
        disp(mean(aps));
        fprintf('~~~~~~~~~~~~~~~~~~~~\n');
        mAP = mean(aps);
    else
        mAP = nan;
    end
else
    %% NMS step
    % binary class, we only extract the 'foreground' class
    raw_aboxes = aboxes{1}; clear aboxes;
    if isempty(opts.nms)
        % normal nms
        best_recall = 0;
        for i = 1:length(nms_overlap_thres)
            
            temp = save_after_nms(1, nms_overlap_thres(i), after_nms_topN);
            if exist(temp, 'file')
                disp('nms result exist. split result (and compute recall) directly ...');
                ld = load(temp);
                aboxes = ld.aboxes; clear ld;
            else
                aboxes = boxes_filter_inline(raw_aboxes, ...
                    per_nms_topN, ...               % -1
                    nms_overlap_thres(i), ...       % 0.6, 0.7, ...
                    after_nms_topN, true);
                save(temp, 'aboxes');
            end
            
            %% =========================
            % make the split result here
            cprintf('blue', 'split the results...\n');
            split_path = [fileparts(temp) '/split'];
            mkdir_if_missing(split_path);
            assert(length(imdb.image_ids) == length(aboxes));
            
            for shit = 1:length(imdb.image_ids)
                boxes = aboxes{shit};
                try
                    save([split_path '/' imdb.image_ids{shit} '.mat'], 'boxes');
                catch
                    mkdir_if_missing(fileparts([split_path '/' imdb.image_ids{shit} '.mat']));
                    save([split_path '/' imdb.image_ids{shit} '.mat'], 'boxes');
                end
            end
            %% =========================
            if compute_recall_switch
                % compute recall
                recall_per_cls = compute_recall_ilsvrc(temp, 300, imdb);
                mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
                
                cprintf('blue', 'nms (thres: %.2f, topN: %d), mean rec:: %.2f\n\n', ...
                    nms_overlap_thres(i), after_nms_topN, mean_recall);
                save([temp(1:end-4) sprintf('_recall_%.2f.mat', mean_recall)], 'recall_per_cls');
                
                if mean_recall > best_recall, best_recall = mean_recall; end
            end
        end
        mAP = best_recall;
    else
        clear aboxes;
        % multi-thres NMS
        parfor kk = 1:length(raw_aboxes)
            aboxes{kk} = AttractioNet_postprocess(raw_aboxes{kk}, 'thresholds', -inf, 'use_gpu', true, ...
                'mult_thr_nms',     true, ...
                'nms_iou_thrs',     opts.nms.nms_iou_thrs, ...
                'factor',           opts.nms.factor, ...
                'scheme',           opts.nms.scheme, ...
                'max_per_image',    opts.nms.max_per_image);
        end
        save([cache_dir_sub '/' opts.nms.note '.mat'], 'aboxes', '-v7.3');
        if compute_recall_switch
            % compute recall
            recall_per_cls = compute_recall_ilsvrc(...
                [cache_dir_sub '/' opts.nms.note '.mat'], 300, imdb);
            mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
            
            cprintf('blue', 'multi-thres nms note (%s), mean rec:: %.2f\n\n', ...
                opts.nms.note, mean_recall);
            save([cache_dir_sub '/' opts.nms.note sprintf('_recall_%.2f.mat', mean_recall)], 'recall_per_cls');
        end
    end
    
end
diary off;
end

function max_rois_num = check_gpu_memory(conf, caffe_net)
%%  try to determine the maximum number of rois
max_rois_num = 0;
for rois_num = 500:500:5000
    % generate pseudo testing data with max size
    im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
    rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
    rois_blob = permute(rois_blob, [3, 4, 1, 2]);
    
    net_inputs = {im_blob, rois_blob};
    
    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    
    caffe_net.forward(net_inputs);
    %     gpuInfo = gpuDevice();
    
    max_rois_num = rois_num;
    
    %     if gpuInfo.FreeMemory < 2 * 10^9  % 2GB for safety
    %         break;
    %     end
end

end

function [boxes, box_inds, thresh] = keep_top_k(boxes, box_inds, end_at, top_k, thresh)
% ------------------------------------------------------------------------
% Keep top K
X = cat(1, boxes{1:end_at});
if isempty(X)
    return;
end
scores = sort(X(:,end), 'descend');
thresh = scores(min(length(scores), top_k));
for image_index = 1:end_at
    if ~isempty(boxes{image_index})
        bbox = boxes{image_index};
        keep = find(bbox(:,end) >= thresh);
        boxes{image_index} = bbox(keep,:);
        box_inds{image_index} = box_inds{image_index}(keep);
    end
end
end
