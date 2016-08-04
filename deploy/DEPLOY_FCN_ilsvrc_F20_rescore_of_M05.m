% RPN training and testing on ilsvrc
%
% refactor by hyli on July 13 2016
% ---------------------------------------------------------

% clc;
clear; close all;
mkdir_if_missing('./deploy/rescore');

ld = load('/media/DATADISK/hyli/project/faster_rcnn/output/rpn_cachedir/M02_s31_stage1_rpn/ilsvrc14_val2/aboxes_filtered_ilsvrc14_val2_final_NMS_0.7.mat');
rpn_proposal = ld.aboxes;
ld = load('/media/DATADISK/hyli/project/faster_rcnn/output/fast_rcnn_cachedir/F05_ls139_nms0_7_top2000_stage1_fast_rcnn/ilsvrc14_val2/final/binary_boxes_ilsvrc14_val2_max_2000_avg_2000.mat');
fcn_proposal = ld.boxes;
% indices of fast-rcnn proposals in the original rpn boxes
inds = ld.inds;

assert(length(fcn_proposal) == length(rpn_proposal));
assert(length(fcn_proposal{1}) == length(rpn_proposal{1}));

new_fcn_boxes = cell(length(fcn_proposal), 1);
new_fcn_boxes_2 = cell(length(fcn_proposal), 1);

for i = 622:length(new_fcn_boxes)
    % per image
    gt_num = min(inds{i})-1;
    rpn_score = rpn_proposal{i}( (inds{i}-gt_num), 5 );
    [new_score, new_ind] = sort(fcn_proposal{i}(:, 5) .* rpn_score, 'descend');
    new_fcn_boxes{i} = [fcn_proposal{i}(new_ind, 1:4) new_score];
    
    [new_score2, new_ind2] = sort(fcn_proposal{i}(:, 5) + rpn_score, 'descend');
    new_fcn_boxes_2{i} = [fcn_proposal{i}(new_ind2, 1:4) new_score2];
end

nms_overlap_thres = [0.8:-0.05:0.4];
imdb.name = 'ilsvrc14_val2';
imdb.flip = false;

for i = 1:length(nms_overlap_thres)
    
    aboxes_mult = boxes_filter_inline(...
        new_fcn_boxes, -1, nms_overlap_thres(i), 2000, true);
    aboxes_add = boxes_filter_inline(...
        new_fcn_boxes_2, -1, nms_overlap_thres(i), 2000, true);
    
    save(['./deploy/rescore/' ...
        sprintf('rescore_boxes_mult_nms_%.2f.mat', nms_overlap_thres(i))], ...
        'aboxes_mult');
    save(['./deploy/rescore/' ...
        sprintf('rescore_boxes_add_nms_%.2f.mat', nms_overlap_thres(i))], ...
        'aboxes_add');
    
    % mult
    recall_per_cls = compute_recall_ilsvrc(...
        ['./deploy/rescore/' sprintf('rescore_boxes_mult_nms_%.2f.mat', nms_overlap_thres(i))], ...
        300, imdb);
    mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
    cprintf('blue', 'nms (thres: %.2f, mult), mean rec:: %.2f\n\n', ...
        nms_overlap_thres(i), mean_recall);
    % add 
    recall_per_cls = compute_recall_ilsvrc(...
        ['./deploy/rescore/' sprintf('rescore_boxes_add_nms_%.2f.mat', nms_overlap_thres(i))], ...
        300, imdb);
    mean_recall = 100*mean(extractfield(recall_per_cls, 'recall'));
    cprintf('blue', 'nms (thres: %.2f, add), mean rec:: %.2f\n\n', ...
        nms_overlap_thres(i), mean_recall);
end







