% neat version of the faster_rcnn_test pipeline

close all; clear;
%clc;
%run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
run('./startup');

caffe_dir = './external/caffe/matlab';
im_path = './datasets/demo';
im_dir = dir([im_path '/*.jpg']);
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_vgg_16layers');
%model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC0712_ZF');

%% init
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 0;    %auto_select_gpu;
opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;
opts.test_scales            = 600;

% load model and configuration
proposal_detection_model = load_proposal_detection_model(model_dir);
proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
proposal_detection_model.conf_proposal.image_means = ...
    gpuArray(proposal_detection_model.conf_proposal.image_means);
proposal_detection_model.conf_detection.image_means = ...
    gpuArray(proposal_detection_model.conf_detection.image_means);

%caffe.init_log(fullfile(pwd, 'caffe_log'));
%active_caffe_mex(opts.gpu_id, opts.caffe_version);
addpath(genpath(caffe_dir));
caffe.reset_all();
caffe.set_device(opts.gpu_id);
caffe.set_mode_gpu();

% proposal net
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
%     [boxes, scores] = proposal_im_detect(proposal_detection_model.conf_proposal, ...
%         rpn_net, im);
%     aboxes = boxes_filter([boxes, scores], opts.per_nms_topN, ...
%         opts.nms_overlap_thres, opts.after_nms_topN);
%
%     if proposal_detection_model.is_share_feature
%         [boxes, scores] = fast_rcnn_conv_feat_detect(...
%             proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
%             rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
%             aboxes(:, 1:4), opts.after_nms_topN);
%     else
%         [boxes, scores] = fast_rcnn_im_detect(...
%             proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
%             aboxes(:, 1:4), opts.after_nms_topN);
%     end
% end

%% -------------------- Actual Eval --------------------
% running_time = [];
classes = proposal_detection_model.classes;
voc_path = '/media/hongyang/research_at_large/Q-dataset/pascal';
dataset = 'VOC2012';    % 'VOC2007' or 'VOC2012'
res_name = 'try1';
subset = 'test';         % 'test' or 'val'
res_name = [res_name '_' subset];
addpath(genpath([voc_path '/VOCdevkit_18-May-2012']));
curr_path = pwd;
cd([voc_path '/VOCdevkit_18-May-2012'])
VOCinit;
cd(curr_path);
thres = 0.6;
evaluate_all_im = 1;

% we test all the val or test images, regardless of the word 'aeroplane'
if strcmp(subset, 'val')
    voc_path_prefix = [voc_path '/VOCdevkit/' dataset];
elseif strcmp(subset, 'test')
    voc_path_prefix = [voc_path '/test_data/VOCdevkit/' dataset];
end
fid = fopen([voc_path_prefix sprintf('/ImageSets/Main/%s_%s.txt', 'aeroplane', subset)]);
list = textscan(fid, '%s %d');
%cls_im_list = list{1}(cellfun(@(x) (x==1), list{2}, 'UniformOutput', false));
if evaluate_all_im
    cls_im_list = list{1};
else
    cls_im_list = list{1}(list{2}==1);
end

mkdir(['local/' dataset '/' res_name]);
fid_cls = zeros(length(classes), 1);
for i = 1:length(classes)
    res_txt = ['local/' dataset '/' res_name ...
        sprintf('/comp3_det_%s_%s.txt', subset, classes{i})];
    fid_cls(i) = fopen(res_txt, 'w');
end

tic;
for j = 1:length(cls_im_list)
    
    im = imread([voc_path_prefix '/JPEGImages/' cls_im_list{j} '.jpg']);
    im = gpuArray(im);
    
    % test proposal
    [boxes, scores] = proposal_im_detect(...
        proposal_detection_model.conf_proposal, rpn_net, im);
    
    aboxes = boxes_filter([boxes, scores], opts.per_nms_topN, ...
        opts.nms_overlap_thres, opts.after_nms_topN);
    
    % test detection
    if proposal_detection_model.is_share_feature
        
        % currently always this case
        blob = rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name);
        config = proposal_detection_model.conf_detection;
        
        [boxes, scores] = fast_rcnn_conv_feat_detect(config, fast_rcnn_net, ...
            im, blob, aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores] = fast_rcnn_im_detect(...
            proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), opts.after_nms_topN);
    end
    
    for mm = 1:length(classes)
        temp_boxes = [boxes(:, (1+(mm-1)*4):(mm*4)), scores(:, mm)];
        temp_boxes = temp_boxes(nms(temp_boxes, 0.3), :);
        %         id = temp_boxes(:, 5) >= thres;
        %         temp_boxes = temp_boxes(id, :);
        
        for kk = 1:size(temp_boxes, 1)
            fprintf(fid_cls(mm), sprintf('%s %f %f %f %f %f\n', ...
                cls_im_list{j}, temp_boxes(kk, 5), temp_boxes(kk, 1:4)));
        end
    end
    if toc > 3
        fprintf('testing %d/%d\n', j, length(cls_im_list));
        tic;
    end
    
end
fclose(fid);

if strcmp(subset, 'val')
    % evaluate now
    cp = sprintf(VOCopts.annocachepath, VOCopts.testset);
    fprintf('pr: loading ground truth...\n');
    load(cp, 'gtids', 'recs');
    gt.gtids = gtids;
    gt.recs = recs;
    
    for mm = 1:length(classes)
        res_txt = ['local/' dataset '/' res_name ...
            sprintf('/comp3_det_val_%s.txt', classes{mm})];
        [recall{mm,1}, prec{mm,1}, ap(mm,1)] = ...
            VOCevaldet_cus(VOCopts, [pwd '/' res_txt], classes{mm}, false, gt);
    end
    
    fprintf('\nmAP: %f\n', mean(ap));
elseif strcmp(subset, 'test')
    fprintf('\ndone! submit to server for evaluation.\n');
end

