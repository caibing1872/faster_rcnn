function [ conf_proposal, conf_fast_rcnn ] = set_config( cache_base_proposal, model )
%SET_CONFIG Summary of this function goes here
%   Detailed explanation goes here

prefix = ['./output/config_temp' '/' cache_base_proposal];
mkdir_if_missing(prefix);

if exist([prefix '/conf_proposal.mat'], 'file')
    v = load([prefix '/conf_proposal.mat']);
    conf_proposal = v.conf_proposal;
    v = load([prefix '/conf_fast_rcnn.mat']);
    conf_fast_rcnn = v.conf_fast_rcnn;
else
    conf_proposal               = proposal_config(model, 'image_means', model.mean_image, 'feat_stride', model.feat_stride);
    conf_fast_rcnn              = fast_rcnn_config('image_means', model.mean_image);
    save([prefix '/conf_proposal.mat'], 'conf_proposal');
    save([prefix '/conf_fast_rcnn.mat'], 'conf_fast_rcnn');
end

