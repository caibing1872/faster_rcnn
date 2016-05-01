clc; clear; close all;

clear is_valid_handle; % to clear init_key
run(fullfile('../', 'startup'));

caffe_dir = '../external/caffe/matlab';
opts.caffe_version          = '';
opts.gpu_id                 = 1;
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;
opts.test_scales            = 600;

model_dir                   = fullfile(pwd, '../output', 'faster_rcnn_final', ...
                                'faster_rcnn_VOC0712_vgg_16layers');
                            
proposal_detection_model    = load_proposal_detection_model(model_dir);
proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;

rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);

%%
load('./val_1000_labels.mat');
load('./k_means_200_converge.mat', 'ctrs2');
im_path = '/home/hongyang/dataset/imagenet_cls/cls_original_image/val';

for i = 1
    
    label = labels{i};
    im = imread([ im_path labels{i}.im_name(end-28:end) ]);
    if ndims(im) == 2
        im = repmat(im,[1,1,3]);
    end
    bbox = convert_bbox(ctrs2, im);
    gtbox = label.gtbox;
%     cls_res = load(sprintf('./wshcls/val_cls_top5/ILSVRC2012_val_%08d.mat',i));
%     cls_res = cls_res.top_5;
    
    %% rpn mask
    [pred_boxes, scores, box_deltas_, anchors_, scores_] = ...
        proposal_im_detect_cus(proposal_detection_model.conf_proposal, ...
        rpn_net, im, proposal_detection_model.image_means);
    
    for k = 7
        M = zeros(size(im,1),size(im,2));
        tyb = pred_boxes(k:9:end, :);
        tys = scores_(k:9:end, :);
        thrs = sort(tys, 'descend');
        thrs = thrs(round(length(thrs)));
        tyb = round(tyb);
        
        for j = 1:size(tyb,1)
            x1 = tyb(j,1);
            y1 = tyb(j,2);
            x2 = tyb(j,3);
            y2 = tyb(j,4);
            M(y1:y2,x1:x2) = M(y1:y2,x1:x2) + (tys(j)>thrs);
        end
    end
    N = M/max(M(:));
    N = N - mean(N(:));
    tim = im.*repmat(uint8(N>0), [1,1,3]);      % masked im
    [mbbox, score] = maskbox(N, bbox, 0);
    mbbox = mbbox(score>0, :);                  % masked box
    
    box_visualize(im, tyb, 2000, 'b', 3);
    
% box_visualize(im, mbbox, 10, 'r', 1);
% box_visualize(im, tyb, 10, 'r', 1);
% box_visualize(im, tyb, 2000, 'r', 1);    
    
end

% box_visualize(im, tyb, 2000, 'r', 0);
% box_visualize(im, tyb, 2000, 'r', 3);
% imshow(N)
% figure; imshow(tim)
% box_visualize(im, mbbox, 10, 'r', 1);
% box_visualize(im, tyb, 10, 'r', 1);
% box_visualize(im, tyb, 2000, 'r', 1);