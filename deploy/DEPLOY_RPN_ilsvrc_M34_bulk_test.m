% RPN testing on ilsvrc
%
% refactor by hyli on July 19 2016
%
% ---------------------------------------------------------

% clc;
clear;
run('./startup');
%% init
% ======================= USER DEFINE =======================

% cache base
cache_base_proposal = 'M34_s31';
gpu_id = 0;
test_proto_name = 'test_9anchor';
train_key = 'train14';

% load paramters from the 'models' folder
model = Model.VGG16_for_Faster_RCNN('solver_10w30w_ilsvrc_9anchor', test_proto_name);
model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, '', model);
% ----------------------------------------------
model.stage1_rpn.nms.note = '0.75';   % must be a string
model.stage1_rpn.nms.nms_overlap_thres = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45];
% ==========================================================

caffe.reset_all();
caffe.set_device(gpu_id);
caffe.set_mode_gpu();

% config
[conf_proposal, conf_fast_rcnn] = Faster_RCNN_Train.set_config( ...
    cache_base_proposal, model, true );
conf_proposal.cache_base_proposal = cache_base_proposal;

% test data
% init:
dataset = [];
% change to point to your devkit install
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path);

%% compute recall and update roidb on TEST
trained_model_dir = ['./output/rpn_cachedir/' cache_base_proposal ...
    '_stage1_rpn/' train_key];
caffemodel_dir = dir([trained_model_dir '/*.caffemodel']);

if strcmp(caffemodel_dir(1).name, 'final.caffemodel')
    
    fprintf('\nComputing final model ...\n');
    RPN_TEST_ilsvrc_hyli(train_key, 'final', ...
        model, dataset, conf_proposal, 'update_roi', false);
    caffemodel_dir = caffemodel_dir(2:end);
end

model_name = extractfield(caffemodel_dir, 'name');
list = cellfun(@(x) str2double(strrep(x(6:end), '.caffemodel', '')), ...
    model_name, 'uniformoutput', false);
list_descend = sort(cell2mat(list'), 'descend');

for i = 1:length(list_descend)
    iter_name = ['iter_' num2str(list_descend(i))];
    fprintf('\nComputing %s model ...\n', iter_name);
    RPN_TEST_ilsvrc_hyli(train_key, iter_name, ...
        model, dataset, conf_proposal, 'update_roi', false);
end

exit;
