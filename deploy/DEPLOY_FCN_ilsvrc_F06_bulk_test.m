% FCN testing on ilsvrc
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
cache_base_rpn = 'M02_s31';
cache_base_fcn = 'F06_ls139_nms0_7_top2000';
train_key = 'train14';
roidb_name = '1';
gpu_id = 0;
binary_train = true;

test_max_per_image          = 2000; %1000; %100;
% if avg == max_per_im, there's no reduce in the number of boxes.
test_avg_per_image          = 2000; %1000; %500; %40;

fast_rcnn_after_nms_topN    = 2000;
fast_nms_overlap_thres = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5];
% ==========================================================

model = Model.VGG16_for_Faster_RCNN(...
    'solver_10w30w_ilsvrc_9anchor', 'test_9anchor', ...     % fixed
    'solver_5w15w_2', 'test_2' ...                          % change here, fast_rcnn
    );
model = Faster_RCNN_Train.set_cache_folder(cache_base_rpn, cache_base_fcn, model);

caffe.reset_all();
caffe.set_device(gpu_id);
caffe.set_mode_gpu();

% config
[~, conf_fast_rcnn] = Faster_RCNN_Train.set_config(cache_base_rpn, model, true);
% dataset
dataset = [];
root_path = './datasets/ilsvrc14_det';
dataset = Dataset.ilsvrc14(dataset, 'test', false, root_path, roidb_name);

%% compute recall and update roidb on TEST
trained_model_dir_prefix = [pwd '/output/fast_rcnn_cachedir/' cache_base_fcn ...
    '_stage1_fast_rcnn/'];
caffemodel_dir = dir([trained_model_dir_prefix train_key '/*.caffemodel']);

if strcmp(caffemodel_dir(1).name, 'final.caffemodel')
    
    cprintf('blue', '\nComputing final model ...\n');   
    fast_rcnn_net_file = [{train_key}, {'final'}];    
    fast_rcnn_test(conf_fast_rcnn, dataset.imdb_test, dataset.roidb_test, ...
        'net_def_file',         model.stage1_fast_rcnn.test_net_def_file, ...
        'net_file',             fast_rcnn_net_file, ...
        'cache_name',           model.stage1_fast_rcnn.cache_name, ...
        'binary',               binary_train, ...
        'max_per_image',        test_max_per_image, ...
        'avg_per_image',        test_avg_per_image, ...
        'nms_overlap_thres',    fast_nms_overlap_thres, ...
        'bulk_prefix',          trained_model_dir_prefix, ...
        'after_nms_topN',       fast_rcnn_after_nms_topN ...
        ); 
    caffemodel_dir = caffemodel_dir(2:end);
end

model_name = extractfield(caffemodel_dir, 'name');
list = cellfun(@(x) str2double(strrep(x(6:end), '.caffemodel', '')), ...
    model_name, 'uniformoutput', false);
list_descend = sort(cell2mat(list'), 'descend');

for i = 1:length(list_descend)
    iter_name = ['iter_' num2str(list_descend(i))];
    cprintf('blue', '\nComputing %s model ...\n', iter_name);
    fast_rcnn_net_file = [{train_key}, {iter_name}];    
    fast_rcnn_test(conf_fast_rcnn, dataset.imdb_test, dataset.roidb_test, ...
        'net_def_file',         model.stage1_fast_rcnn.test_net_def_file, ...
        'net_file',             fast_rcnn_net_file, ...
        'cache_name',           model.stage1_fast_rcnn.cache_name, ...
        'binary',               binary_train, ...
        'max_per_image',        test_max_per_image, ...
        'avg_per_image',        test_avg_per_image, ...
        'bulk_prefix',          trained_model_dir_prefix, ...
        'nms_overlap_thres',    fast_nms_overlap_thres, ...
        'after_nms_topN',       fast_rcnn_after_nms_topN ...
        );
end

exit;
