% neat version of the faster_rcnn_test pipeline

close all; clear;
%clc;
%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
run('./startup');

caffe_dir = './external/caffe/matlab';
im_path = './datasets/demo';
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers');
% model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_ZF');

%% init
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 0;
%auto_select_gpu;
opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
%opts.use_gpu                = true;
%opts.test_scales            = 600;

addpath(genpath(caffe_dir));
%active_caffe_mex(opts.gpu_id, opts.caffe_version);
im_dir = dir([im_path '/*.jpg']);
proposal_detection_model = load_proposal_detection_model(model_dir);

% uncomment the following if you don't change the type of image_mean to GPU
% in the model.mat file
% proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
% proposal_detection_model.conf_detection.test_scales = opts.test_scales;
% proposal_detection_model.conf_proposal.image_means = ...
%     gpuArray(proposal_detection_model.conf_proposal.image_means);
% proposal_detection_model.conf_detection.image_means = ...
%     gpuArray(proposal_detection_model.conf_detection.image_means);

% caffe.init_log(fullfile(pwd, 'caffe_log'));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();
%proposal net
rpn_net = caffe.Net(proposal_detection_model.proposal_net_def, 'test');
rpn_net.copy_from(proposal_detection_model.proposal_net);
% fast rcnn net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

% for j = 1:2 % we warm up 2 times
%     im = uint8(ones(375, 500, 3)*128);
%     if opts.use_gpu
%         im = gpuArray(im);
%     end
%     [boxes, scores]             = proposal_im_detect(proposal_detection_model.conf_proposal, rpn_net, im);
%     aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
%     if proposal_detection_model.is_share_feature
%         [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
%             rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
%             aboxes(:, 1:4), opts.after_nms_topN);
%     else
%         [boxes, scores]             = fast_rcnn_im_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
%             aboxes(:, 1:4), opts.after_nms_topN);
%     end
% end

% running_time = [];
for j = 1:length(im_dir)
    
    im = imread([im_path '/' im_dir(j).name]);
    im = gpuArray(im);
    
    % test proposal
    th = tic();
    [boxes, scores] = proposal_im_detect(...
        proposal_detection_model.conf_proposal, rpn_net, im);
    t_proposal = toc(th);
    
    th = tic();
    aboxes = boxes_filter([boxes, scores], opts.per_nms_topN, ...
        opts.nms_overlap_thres, opts.after_nms_topN);
    t_nms = toc(th);
    
    % test detection
    th = tic();
    if proposal_detection_model.is_share_feature
        % currently always this case
        blob = rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name);
        config = proposal_detection_model.conf_detection;
        [boxes, scores] = fast_rcnn_conv_feat_detect(config, fast_rcnn_net, im, blob, aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores] = fast_rcnn_im_detect(...
            proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
    t_detection = toc(th);
    
%     fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', ...
%         im_dir(j).name, size(im, 2), size(im, 1), ...
%         t_proposal + t_nms + t_detection, t_proposal, t_nms+t_detection);
%     running_time(end+1) = t_proposal + t_nms + t_detection;
    
    % visualize
    classes = proposal_detection_model.classes;
    boxes_cell = cell(length(classes), 1);
    thres = 0.6;
    for i = 1:length(boxes_cell)
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
        
        I = boxes_cell{i}(:, 5) >= thres;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end
    figure(j);
    showboxes(im, boxes_cell, classes, 'voc');
    pause(0.1);
end
% fprintf('mean time: %.3fs\n', mean(running_time));
