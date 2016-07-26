function save_model_path = fast_rcnn_train(conf, imdb_train, roidb_train, train_key, varargin)
% save_model_path = fast_rcnn_train(conf, imdb_train, roidb_train, varargin)
% --------------------------------------------------------
% Fast R-CNN
% Reimplementation based on Python Fast R-CNN (https://github.com/rbgirshick/fast-rcnn)
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

%% inputs
ip = inputParser;
ip.addRequired('conf',                                  @isstruct);
ip.addRequired('imdb_train',                            @iscell);
ip.addRequired('roidb_train',                           @iscell);
ip.addRequired('train_key',                             @isstr);

ip.addParameter('do_val',               false,          @isscalar);
ip.addParameter('imdb_val',             struct(),       @isstruct);
ip.addParameter('roidb_val',            struct(),       @isstruct);
ip.addParameter('val_iters',            500,            @isscalar);
ip.addParameter('val_interval',         2000,           @isscalar);
ip.addParameter('snapshot_interval',    10000,          @isscalar);
ip.addParameter('solver_def_file',      '',             @isstr);
ip.addParameter('net_file',             '',             @isstr);
ip.addParameter('cache_name',           'Zeiler_conv5', @isstr);
ip.addParameter('debug',                false,          @isscalar);
ip.addParameter('solverstate',          '',             @isstr);
ip.addParameter('binary',               true,           @islogical);
ip.parse(conf, imdb_train, roidb_train, train_key, varargin{:});
opts = ip.Results;

debug = opts.debug;
%% try to find trained model
%imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
cache_dir = fullfile(pwd, 'output', 'fast_rcnn_cachedir', opts.cache_name, opts.train_key);
save_model_path = fullfile(cache_dir, 'final');
if exist(save_model_path, 'file')
    return;
end

%% init
% init caffe log
mkdir_if_missing([cache_dir '/caffe_log']);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log/train_');
caffe.init_log(caffe_log_file_base);
% init matlab log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'matlab_log'));
log_file = fullfile(cache_dir, 'matlab_log', ['train_', timestamp, '.txt']);
diary(log_file);

% init caffe solver
caffe_solver = caffe.Solver(opts.solver_def_file);
caffe_solver.net.copy_from(opts.net_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

disp('conf:');
disp(conf);
disp('opts:');
disp(opts);

%% making tran/val data
mkdir_if_missing('./output/training_test_data/');
% training
train_data_name = ['FCN_train_' conf.cache_base_proposal];
if exist(sprintf('./output/training_test_data/%s.mat', train_data_name), 'file')
    
    ld = load(sprintf('./output/training_test_data/%s.mat', train_data_name));
    image_roidb_train = ld.image_roidb_train;
    bbox_means = ld.bbox_means;
    bbox_stds = ld.bbox_stds;
    fprintf('Loading existant FCN training data (%s) ...', train_data_name);
    clear ld;
    fprintf(' Done.\n');
else
    fprintf('Preparing FCN training data (%s) ...\n', train_data_name);
    [image_roidb_train, bbox_means, bbox_stds]...
        = fast_rcnn_prepare_image_roidb(conf, opts.imdb_train, opts.roidb_train);
    save(sprintf('./output/training_test_data/%s.mat', train_data_name), ...
        'image_roidb_train', 'bbox_means', 'bbox_stds', '-v7.3');
    fprintf(' Done and saved.\n\n');
end

% validation
val_data_name = ['FCN_val_' conf.cache_base_proposal];
if opts.do_val
    
    if exist(sprintf('./output/training_test_data/%s.mat', val_data_name), 'file')
        
        ld = load(sprintf('./output/training_test_data/%s.mat', val_data_name));
        fprintf('Loading existant FCN validation data (%s) ...', val_data_name);
        image_roidb_val = ld.image_roidb_val;
        shuffled_inds_val = ld.shuffled_inds_val;
        clear ld;
        fprintf(' Done.\n');
    else
        fprintf('Preparing FCN validation data (%s) ...\n', val_data_name);
        [image_roidb_val]...
            = fast_rcnn_prepare_image_roidb(conf, opts.imdb_val, opts.roidb_val, bbox_means, bbox_stds);
        % fix validation data
        shuffled_inds_val = generate_random_minibatch([], image_roidb_val, conf.ims_per_batch);
        shuffled_inds_val = shuffled_inds_val(randperm(length(shuffled_inds_val), opts.val_iters));
        
        save(sprintf('./output/training_test_data/%s.mat', val_data_name), ...
            'image_roidb_val', 'shuffled_inds_val', '-v7.3');
        fprintf(' Done and saved.\n\n');
    end
end

% try to train/val with images which have maximum size potentially,
% to validate whether the gpu memory is enough
num_classes = size(image_roidb_train(1).overlap, 2);
if opts.binary, num_classes = 1; end
check_gpu_memory(conf, caffe_solver, num_classes, opts.do_val);

%% training
shuffled_inds = [];
train_results = [];
train_res_total = [];
val_results = [];
iter_ = caffe_solver.iter();
max_iter = caffe_solver.max_iter();
th = tic;

while (iter_ < max_iter)
    
    caffe_solver.net.set_phase('train');
    % generate minibatch training data
    [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, conf.ims_per_batch);
    [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
        fast_rcnn_get_minibatch_binary(conf, image_roidb_train(sub_db_inds));
    
    net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
    caffe_solver.net.reshape_as_input(net_inputs);
    
    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    rst = caffe_solver.net.get_output();
    train_results = parse_rst(train_results, rst);
    train_res_total = parse_rst(train_res_total, rst);
    
    if debug && ~mod(iter_, 20)
        fprintf('iter: %d\n', iter_)
        %check_loss(rst, caffe_solver, net_inputs);
        fprintf('\n');
    end
    
    % do valdiation per val_interval iterations
    if ~mod(iter_, opts.val_interval)
        if opts.do_val
            caffe_solver.net.set_phase('test');
            for i = 1:length(shuffled_inds_val)
                sub_db_inds = shuffled_inds_val{i};
                [im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob] = ...
                    fast_rcnn_get_minibatch(conf, image_roidb_val(sub_db_inds));
                
                % Reshape net's input blobs
                net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};
                caffe_solver.net.reshape_as_input(net_inputs);
                
                caffe_solver.net.forward(net_inputs);
                
                rst = caffe_solver.net.get_output();
                val_results = parse_rst(val_results, rst);
            end
        end
        
        show_state(iter_, train_results, val_results);
        train_results = [];
        %val_results = [];
        diary; diary; % flush diary
    end
    
    % snapshot
    if ~mod(iter_, opts.snapshot_interval)
        snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        save([cache_dir '/' sprintf('loss_%d.mat', iter_)], 'train_res_total', 'val_results');
    end
    % training progress report
    if ~mod(iter_, 100)
        time = toc(th);
        fprintf('iter %d, loss %.4f, time: %.2f min, estTime: %.2f hour\n', ...
            iter_, (10*rst(2).data + rst(3).data), time/60, (time/3600)*(max_iter-iter_)/100);
        th = tic;
    end
    iter_ = caffe_solver.iter();
end

% final snapshot
snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
save_model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');
save([cache_dir '/' sprintf('loss_final_iter_%d.mat', max_iter)], 'train_res_total', 'val_results');

diary off;
caffe.reset_all();
rng(prev_rng);
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train, ims_per_batch)

% shuffle training data per batch
if isempty(shuffled_inds)
    % make sure each minibatch, only has horizontal images or vertical
    % images, to save gpu memory
    
    hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
    vert_image_inds = ~hori_image_inds;
    hori_image_inds = find(hori_image_inds);
    vert_image_inds = find(vert_image_inds);
    
    % random perm
    lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
    hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
    lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
    vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
    
    % combine sample for each ims_per_batch
    hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
    vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
    
    shuffled_inds = [hori_image_inds, vert_image_inds];
    shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
    
    shuffled_inds = num2cell(shuffled_inds, 1);
end

if nargout > 1
    % generate minibatch training data
    sub_inds = shuffled_inds{1};
    assert(length(sub_inds) == ims_per_batch);
    shuffled_inds(1) = [];
end
end

function check_gpu_memory(conf, caffe_solver, num_classes, do_val)
%%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough

% generate pseudo training data with max size
im_blob = single(zeros(max(conf.scales), conf.max_size, 3, conf.ims_per_batch));
rois_blob = single(repmat([0; 0; 0; max(conf.scales)-1; conf.max_size-1], 1, conf.batch_size));
rois_blob = permute(rois_blob, [3, 4, 1, 2]);
labels_blob = single(ones(conf.batch_size, 1));
labels_blob = permute(labels_blob, [3, 4, 2, 1]);
bbox_targets_blob = zeros(4 * (num_classes+1), conf.batch_size, 'single');
bbox_targets_blob = single(permute(bbox_targets_blob, [3, 4, 1, 2]));
bbox_loss_weights_blob = bbox_targets_blob;

net_inputs = {im_blob, rois_blob, labels_blob, bbox_targets_blob, bbox_loss_weights_blob};

% Reshape net's input blobs
caffe_solver.net.reshape_as_input(net_inputs);

% one iter SGD update
caffe_solver.net.set_input_data(net_inputs);
caffe_solver.step(1);

if do_val
    % use the same net with train to save memory
    caffe_solver.net.set_phase('test');
    caffe_solver.net.forward(net_inputs);
    caffe_solver.net.set_phase('train');
end
end

function model_path = snapshot(caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
bbox_stds_flatten = reshape(bbox_stds', [], 1);
bbox_means_flatten = reshape(bbox_means', [], 1);

% merge bbox_means, bbox_stds into the model
bbox_pred_layer_name = 'bbox_pred';
weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
weights_back = weights;
biase_back = biase;

weights = ...
    bsxfun(@times, weights, bbox_stds_flatten'); % weights = weights * stds;
biase = ...
    biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

model_path = [fullfile(cache_dir, file_name) '.caffemodel'];
caffe_solver.net.save(model_path);
fprintf('Saved as %s\n', [file_name '.caffemodel']);
solverstate_path = [fullfile(cache_dir, file_name) '.solverstate'];
caffe_solver.snapshot(solverstate_path, model_path);
fprintf('Saved as %s\n', [file_name '.solverstate']);

% restore net to original state
caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);
end

function show_state(iter, train_results, val_results)
fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
fprintf('Training : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
    1 - mean(train_results.accuarcy.data), ...
    mean(train_results.loss_cls.data), ...
    mean(train_results.loss_bbox.data));
if exist('val_results', 'var') && ~isempty(val_results)
    fprintf('Testing  : error %.3g, loss (cls %.3g, reg %.3g)\n', ...
        1 - mean(val_results.accuarcy.data), ...
        mean(val_results.loss_cls.data), ...
        mean(val_results.loss_bbox.data));
end
end
