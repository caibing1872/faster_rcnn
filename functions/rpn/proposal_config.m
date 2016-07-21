function conf = proposal_config(model, varargin)
% conf = proposal_config(varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

% note: if new params are added, delete files in 'output/config_temp'
ip = inputParser;
%% training
% deprecated
ip.addParameter('use_chunk_if_train_data_large', true, @islogical);

ip.addParameter('use_gpu',          gpuDeviceCount > 0, @islogical);
% whether drop the anchors that has edges outside of the image boundary
ip.addParameter('drop_boxes_runoff_image', true, @islogical);
% Image scales -- the short edge of input image
ip.addParameter('scales',           600,            @ismatrix);
% Max pixel size of a scaled input image
ip.addParameter('max_size',         1000,           @isscalar);
% Images per batch, only supports ims_per_batch = 1 currently
ip.addParameter('ims_per_batch',    1,              @isscalar);
% Minibatch size
ip.addParameter('batch_size',       256,            @isscalar);
% Fraction of minibatch that is foreground labeled (class > 0)
ip.addParameter('fg_fraction',      0.5,            @isscalar);
% weight of background samples, when weight of foreground samples is
% 1.0
ip.addParameter('bg_weight',        1.0,            @isscalar);
% Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
ip.addParameter('fg_thresh',        0.7,            @isscalar);
% Overlap threshold for a ROI to be considered background (class = 0 if
% overlap in [bg_thresh_lo, bg_thresh_hi))
ip.addParameter('bg_thresh_hi',     0.3,            @isscalar);
ip.addParameter('bg_thresh_lo',     0,              @isscalar);
% mean image, in RGB order
ip.addParameter('image_means',      128,            @ismatrix);
% Use horizontally-flipped images during training?
ip.addParameter('use_flipped',      true,           @islogical);
% Stride in input image pixels at ROI pooling level (network specific)
% 16 is true for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
ip.addParameter('feat_stride',      16,             @isscalar);

% train proposal target only to labled ground-truths or also include
% other proposal results (selective search, etc.)
% affect in generating the training samples
% ('proposal_prepare_image_roidb.m')
ip.addParameter('target_only_gt',   true,           @islogical);

% random seed
ip.addParameter('rng_seed',         6,              @isscalar);

% scale of anchor size
% default
ip.addParameter('anchor_scale',     2.^[3:5],       @ismatrix);
ip.addParameter('ratios',           [0.5, 1, 2],    @ismatrix);

%% testing
ip.addParameter('test_scales',          600,            @isscalar);
ip.addParameter('test_max_size',        1000,           @isscalar);
%ip.addParameter('test_nms',             0.3,            @isscalar);
ip.addParameter('test_binary',          false,          @islogical);
ip.addParameter('test_min_box_size',    16,             @isscalar);
ip.addParameter('test_drop_boxes_runoff_image', ...
                                        false,          @islogical);
ip.parse(varargin{:});
conf = ip.Results;
assert(conf.ims_per_batch == 1, 'currently rpn only supports ims_per_batch == 1');

% if image_means is a file, load it
if ischar(conf.image_means)
    s = load(conf.image_means);
    s_fieldnames = fieldnames(s);
    assert(length(s_fieldnames) == 1);
    conf.image_means = s.(s_fieldnames{1});
end

%% added by hyli
% move the following from main function
% generate anchors and pre-calculate output size of rpn network
% [conf.anchors, conf.output_width_map, conf.output_height_map] ...
%     = proposal_prepare_anchors(conf, model.stage1_rpn.cache_name, ...
%     model.stage1_rpn.test_net_def_file);

[conf.output_width_map, conf.output_height_map] = ...
    proposal_calc_output_size(conf, model.stage1_rpn.test_net_def_file);

conf.anchors = proposal_generate_anchors(model.stage1_rpn.cache_name, ...
    'scales',   conf.anchor_scale, ...
    'ratios',   conf.ratios ...
    );
end