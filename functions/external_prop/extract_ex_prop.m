clear; close all;
nms_range = [.8 : -.05 : 0.3];

%result_name = 'aug_5th_edge';
%method = 'edgebox';
%result_name = 'aug_5th_ss';
%method = 'selective_search';

%sub_dataset = 'val1';
%imdb.name = 'ilsvrc14_val1';
%sub_dataset = 'train14';
%imdb.name = 'ilsvrc14_train14';

%sub_dataset = 'real_test';
%sub_dataset = 'val1_14';
sub_dataset = 'val1_13';
%sub_dataset = 'pos1k_13';
result_name = 'edgebox';
%result_name = 'ss';
%result_name = 'attractioNet';

imdb.name = sprintf('ilsvrc14_%s', sub_dataset);
% note: we don't differentiate top_k when saving them
% top_k = [300, 500, 1000, 2000];
top_k = 300;
method = result_name;
%% config
addpath(genpath('./functions/external_prop'));
mkdir_if_missing(['./box_proposals/' sub_dataset '/']);
mkdir_if_missing(['./box_proposals/' sub_dataset '/' result_name]);
save_name = ['./box_proposals/' sub_dataset '/' result_name '/boxes_right_format.mat'];

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

%% extract boxes
if strcmp(method, 'edgebox')
    
    if ~exist(save_name, 'file')
        aboxes = cell(length(test_im_list), 1);
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
        
        for i = 1:length(test_im_list)
            if i == 1 || i == length(test_im_list) || mod(i, 1000) == 0
                fprintf('extract box, method: %s, dataset: %s, (%d/%d)...\n', ...
                    method, sub_dataset, i, length(test_im_list));
            end
            im = imread([im_path '/' test_im_list{i} extension]);
            if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
            temp = edgeBoxes(im, model, opts);
            % [x, y, w, h, score]
            temp(:, 3) = temp(:, 1) + temp(:, 3) - 1;
            temp(:, 4) = temp(:, 2) + temp(:, 4) - 1;
            aboxes{i} = temp;
        end
        save(save_name, 'aboxes', '-v7.3');
        fprintf('\ndone saving the original boxes (%s)!\n', save_name);
    end
    
elseif strcmp(method, 'ss')
    
    if ~exist(save_name, 'file')
        aboxes = cell(length(test_im_list), 1);
        for i = 1:length(test_im_list)
            if i == 1 || i == length(test_im_list) || mod(i, 1000) == 0
                fprintf('extract box, method: %s, dataset: %s, (%d/%d)...\n', ...
                    method, sub_dataset, i, length(test_im_list));
            end
            im = imread([im_path '/' test_im_list{i} extension]);
            if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
            [temp, score] = selective_search_boxes(im);
            % [y1 x1 y2 x2]
            temp = temp(:, [2 1 4 3]);
            aboxes{i} = [temp, score];
        end
        save(save_name, 'aboxes', '-v7.3');
    end
end

%% compute recall
for i = 1:length(top_k)
    
    if ~isempty(nms_range)
        % try different nms
        ld = load(save_name);
        aboxes_raw = ld.aboxes;
        % make sure it has a score
        assert(size(aboxes_raw{1}, 2)==5);
        
        for j = 1:length(nms_range)
            
            aboxes = boxes_filter_inline(aboxes_raw, -1, nms_range(j), -1, true);
            save([save_name(1:end-4) sprintf('_nms_%.2f.mat', nms_range(j))], ...
                'aboxes', '-v7.3');
            
            recall_per_cls = compute_recall_ilsvrc(...
                [save_name(1:end-4) sprintf('_nms_%.2f.mat', nms_range(j))], top_k(i), imdb);
            mean_recall = mean(extractfield(recall_per_cls, 'recall'));
            cprintf('blue', 'method:: %s, top_k:: %d, nms:: %.2f, mean rec:: %.2f\n\n', ...
                method, top_k(i), nms_range(j), 100*mean_recall);
        end
    else
        % no nms
        recall_per_cls = compute_recall_ilsvrc(save_name, top_k(i), imdb);
        mean_recall = mean(extractfield(recall_per_cls, 'recall'));
        cprintf('blue', 'method:: %s, top_k:: %d, mean rec:: %.2f\n\n', ...
            method, top_k(i), 100*mean_recall);
    end
end
disp('done!');
exit;
