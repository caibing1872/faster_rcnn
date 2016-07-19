function aboxes = proposal_test(conf, imdb, varargin)

ip = inputParser;
ip.addRequired('conf',                              @isstruct);
ip.addRequired('imdb',                              @isstruct);

ip.addParameter('net_def_file',                     @isstr);
ip.addParameter('net_file',                         @isstr);
ip.addParameter('cache_name',                       @isstr);
ip.addParameter('suffix',          '',              @isstr);

ip.parse(conf, imdb, varargin{:});
opts = ip.Results;

cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', opts.cache_name, imdb.name);
%% init net
% init caffe log
mkdir_if_missing([cache_dir '/caffe_log']);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log/test_');
caffe.init_log(caffe_log_file_base);
% init caffe net
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

% init matlab log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'matlab_log'));
log_file = fullfile(cache_dir, 'matlab_log', ['test_', timestamp, '.txt']);
diary(log_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

disp('opts:');
disp(opts);
disp('conf:');
disp(conf);

%% testing
num_images = length(imdb.image_ids);
% all detections are collected into:
%    all_boxes[image] = N x 5 array of detections in
%    (x1, y1, x2, y2, score)
aboxes = cell(num_images, 1);
abox_deltas = cell(num_images, 1);
aanchors = cell(num_images, 1);
ascores = cell(num_images, 1);

count = 0;
show_num = 3000;
for i = 1:num_images
    count = count + 1;
%     tic_toc_print('%s: test %s (%s) %d/%d \n', procid(), ...
%         opts.suffix, imdb.name, count, num_images);
    if i == 1 || i == length(test_im_list) || mod(i, show_num)==0
        fprintf('%s: test %s (%s) %d/%d \n', procid(), ...
            opts.suffix, imdb.name, count, num_images);
    end
    %th = tic;
    im = imread(imdb.image_at(i));
    [boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect(conf, caffe_net, im);
    
    %fprintf(' time: %.3fs\n', toc(th)); 
    aboxes{i} = [boxes, scores];
end
%save(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes', '-v7.3');

diary off;
caffe.reset_all();
rng(prev_rng);
end
