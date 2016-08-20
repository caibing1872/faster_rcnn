clear; close all;
nms_range = [.8 : -.05 : 0.3];

%sub_dataset = 'real_test';
%sub_dataset = 'val1_14';
%sub_dataset = 'val1_13';
sub_dataset = 'pos1k_13';

%result_name = 'edgebox';
result_name = 'ss';
fucking_start_im = 1; %125001;
%fucking_end_im = 100000; %length(test_im_list);

imdb.name = sprintf('ilsvrc14_%s', sub_dataset);
% note: we don't differentiate top_k when saving them
% top_k = [300, 500, 1000, 2000];
top_k = 300;
method = result_name;
%% config
addpath(genpath('./functions/external_prop'));
save_name = ['./box_proposals/' sub_dataset '/' result_name '/split'];
mkdir_if_missing(save_name);

switch imdb.name
    case 'pascal'
        % pascal test set
        % not tested since update to ilsvrc dataset
        dataset_root = '/media/hongyang/research_at_large/Q-dataset/pascal/VOCdevkit/VOC2007';
        testset = [dataset_root '/ImageSets/Main/*_test.txt'];
        testset_dir = dir(testset);
        fid = fopen([dataset_root '/ImageSets/Main/' testset_dir(1).name], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [dataset_root '/JPEGImages'];
        extension = '.jpg';
        
    case 'ilsvrc14_train14'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/train14.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2014_DET_train'];
        extension = '.JPEG';
        imdb.flip = true;
        
    case 'ilsvrc14_val1'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
        imdb.flip = true;
        
    case 'ilsvrc14_val2'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
        imdb.flip = false;
        
        
        % the following datasets wont compute recall since I am just too lazy
        % to collect their GT info.
    case 'ilsvrc14_val1_14'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1_14.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2014_DET_train'];
        extension = '';
        imdb.flip = false;
        
    case 'ilsvrc14_val1_13'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val1_13.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
        imdb.flip = false;
        
    case 'ilsvrc14_real_test'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/real_test.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2015_DET_test'];
        extension = '.JPEG';
        imdb.flip = false;
        
    case 'ilsvrc14_pos1k_13'
        root_folder = './datasets/ilsvrc14_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/pos1k_13.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2014_DET_train'];
        extension = '.JPEG';
        imdb.flip = false;
end
if imdb.flip
    test_im_list_flip = cellfun(@(x) [x '_flip'], test_im_list, 'uniformoutput', false);
    test_im_list_new = cell(2*length(test_im_list_flip), 1);
    test_im_list_new(1:2:end) = test_im_list;
    test_im_list_new(2:2:end) = test_im_list_flip;
    test_im_list = test_im_list_new;
end
if ~exist('fucking_end_im', 'var'), fucking_end_im = length(test_im_list); end

%% extract boxes
if strcmp(method, 'edgebox')
    
    model = load('edgebox/models/forest/modelBsds');
    model = model.model;
    model.opts.multiscale=0;
    model.opts.sharpen=2;
    model.opts.nThreads=4;
    % set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .65;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e4;  % max number of boxes to detect
    
    parfor i = fucking_start_im : fucking_end_im
%         if i == fucking_start_im || i == fucking_end_im || mod(i, 1000) == 0
%             fprintf('extract box, method: %s, dataset: %s, %d / (%d-%d)...\n', ...
%                 method, sub_dataset, i, fucking_start_im, fucking_end_im);
%         end
        if ~exist([save_name '/' test_im_list{i}(11:end) '.mat'], 'file')
            im = imread([im_path '/' test_im_list{i} extension]);
            if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
            boxes_edge = edgeBoxes(im, model, opts);
            % [x, y, w, h, score]
            boxes_edge(:, 3) = boxes_edge(:, 1) + boxes_edge(:, 3) - 1;
            boxes_edge(:, 4) = boxes_edge(:, 2) + boxes_edge(:, 4) - 1;
            parsave([save_name '/' test_im_list{i}(11:end) '.mat'], boxes_edge);
        end
    end
    
elseif strcmp(method, 'ss')

    parfor i = fucking_start_im : fucking_end_im
%         if i == fucking_start_im || i == fucking_end_im || mod(i, 1000) == 0
%             fprintf('extract box, method: %s, dataset: %s, %d / (%d-%d), total: (%d) ...\n', ...
%                 method, sub_dataset, i, fucking_start_im, fucking_end_im, length(test_im_list));
%         end
        if ~exist([save_name '/' test_im_list{i}(11:end) '.mat'], 'file')
%             if i == 102065 || i == 102087
%                 boxes = [];
%                 save([save_name '/' test_im_list{i}(11:end) '.mat'], 'boxes');
%                 continue;
%             end          
            im = imread([im_path '/' test_im_list{i} extension]);            
            if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
            try
                [temp, score] = selective_search_boxes(im);
                % [y1 x1 y2 x2]
                temp = temp(:, [2 1 4 3]);
                boxes_ss = [temp, score];
            catch
                boxes_ss = [];
            end
            parsave([save_name '/' test_im_list{i}(11:end) '.mat'], boxes_ss);
        end
    end
end

%% compute recall
% first merge the fucking results together
if (length(test_im_list) ~= length(unique(test_im_list))), warning('list is not unique'); end

im_num = length(dir([save_name '/*.mat']));
assert(im_num == length(unique(test_im_list)), ...
    sprintf('fuck! actual no of images vs total supposed no: %d vs %d', ...
    im_num, length(unique(test_im_list))));

save_name_new = [save_name '/../boxes_right_format.mat'];
if ~exist(save_name_new, 'file')
    fprintf('merge these split results\n\n');
    aboxes = cell(length(test_im_list), 1);
    for i = 1:length(test_im_list)
        load([save_name '/' test_im_list{i}(11:end) '.mat']);
        aboxes{i} = boxes;
    end
    save(save_name_new, 'aboxes', '-v7.3');
end

for i = 1:length(top_k)
    
    if ~isempty(nms_range)
        % try different nms
        ld = load(save_name_new);
        aboxes_raw = ld.aboxes;
        % make sure it has a score
        assert(size(aboxes_raw{1}, 2)==5);
        
        for j = 1:length(nms_range)
            
            aboxes = boxes_filter_inline(aboxes_raw, -1, nms_range(j), -1, true);
            save([save_name_new(1:end-4) sprintf('_nms_%.2f.mat', nms_range(j))], ...
                'aboxes', '-v7.3');
            recall_per_cls = compute_recall_ilsvrc(...
            
            if recall_per_cls > 0
                mean_recall = mean(extractfield(recall_per_cls, 'recall'));
                cprintf('blue', 'method:: %s, top_k:: %d, nms:: %.2f, mean rec:: %.2f\n\n', ...
                    method, top_k(i), nms_range(j), 100*mean_recall);
            end
        end
    else
        % no nms
        recall_per_cls = compute_recall_ilsvrc(save_name_new, top_k(i), imdb);
        mean_recall = mean(extractfield(recall_per_cls, 'recall'));
        cprintf('blue', 'method:: %s, top_k:: %d, mean rec:: %.2f\n\n', ...
            method, top_k(i), 100*mean_recall);
    end
end
disp('done!');
exit;
