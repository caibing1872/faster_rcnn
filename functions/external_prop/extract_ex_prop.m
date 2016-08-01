%clc;
clear; close all;

nms_range = [.8 : -.05 : 0.3];
%nms_range = [.45 : -.05 : 0.3];
% result_name = 'aug_1st_edge';
% method = 'edgebox';
update_edge_format = false;

result_name = 'aug_1st_ss_score';
method = 'selective_search';
update_ss_format = false;

imdb.name = 'ilsvrc14_val2';
imdb.flip = false;
dataset = 'ilsvrc';

%top_k = [300, 500, 1000, 2000];
top_k = 300;
%%
addpath(genpath('./functions/external_prop/edgebox'));
addpath(genpath('./functions/external_prop/selective_search'));
result_path = './box_proposals/val2';
mkdir_if_missing([result_path '/' result_name]);
save_name = [result_path '/' result_name '/boxes.mat'];
switch dataset
    case 'pascal'
        % pascal test set
        dataset_root = '/media/hongyang/research_at_large/Q-dataset/pascal/VOCdevkit/VOC2007';
        testset = [dataset_root '/ImageSets/Main/*_test.txt'];
        testset_dir = dir(testset);
        fid = fopen([dataset_root '/ImageSets/Main/' testset_dir(1).name], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [dataset_root '/JPEGImages'];
        extension = '.jpg';
    case 'ilsvrc'
        % ilsvrc val2
        root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
        fid = fopen([root_folder '/data/det_lists/val2.txt'], 'r');
        temp = textscan(fid, '%s%s');
        test_im_list = temp{1}; clear temp;
        im_path = [root_folder '/../ILSVRC2013_DET_val'];
        extension = '.JPEG';
end

%%
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
        
        parfor i = 1:length(test_im_list)
            im = imread([im_path '/' test_im_list{i} extension]);
            if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
            temp = edgeBoxes(im, model, opts);
            % [x, y, w, h, score]
            temp(:, 3) = temp(:, 1) + temp(:, 3) - 1;
            temp(:, 4) = temp(:, 2) + temp(:, 4) - 1;
            aboxes{i} = temp;
        end
        save(save_name, 'aboxes');
    end
    
elseif strcmp(method, 'selective_search')
    
    if ~exist(save_name, 'file')
        aboxes = cell(length(test_im_list), 1);
        parfor i = 1:length(test_im_list)
            im = imread([im_path '/' test_im_list{i} extension]);
            if size(im, 3) == 1, im = repmat(im, [1 1 3]); end
            [temp, score] = selective_search_boxes(im);
            % [y1 x1 y2 x2]
            temp = temp(:, [2 1 4 3]);
            aboxes{i} = [temp, score];
        end
        save(save_name, 'aboxes');
    end
end

% almost deprecated below
if update_edge_format
    
    load(save_name);
    aboxes = cellfun(@change_edgebox, aboxes, 'uniformoutput', false);
    save([save_name(1:end-4) '_right_format.mat'], 'aboxes');
end
if update_ss_format
    
    load(save_name);
    aboxes = cellfun(@(x) x(:, [2 1 4 3]), aboxes, 'uniformoutput', false);
    save([save_name(1:end-4) '_right_format.mat'], 'aboxes');
end

%%
for i = 1:length(top_k)
    
    if ~isempty(nms_range)
        
        ld = load([save_name(1:end-4) '_right_format.mat']);
        aboxes_raw = ld.aboxes;
        assert(size(aboxes_raw{1}, 2)==5);
        for j = 1:length(nms_range)
            aboxes = boxes_filter_inline(aboxes_raw, -1, nms_range(j), 2000, true);
            save([save_name(1:end-4) sprintf('_nms_%.2f.mat', nms_range(j))], ...
                'aboxes', '-v7.3');
            
            recall_per_cls = compute_recall_ilsvrc(...
                [save_name(1:end-4) sprintf('_nms_%.2f.mat', nms_range(j))], top_k(i), imdb);
            mean_recall = mean(extractfield(recall_per_cls, 'recall'));
            cprintf('blue', 'method:: %s, top_k:: %d, nms:: %.2f, mean rec:: %.2f\n\n', ...
                method, top_k(i), nms_range(j), 100*mean_recall);
        end
    else
        recall_per_cls = compute_recall_ilsvrc(...
            [save_name(1:end-4) '_right_format.mat'], top_k(i), imdb);
        %     recall_per_cls = compute_recall_ilsvrc(save_name, top_k(i), imdb);
        
        mean_recall = mean(extractfield(recall_per_cls, 'recall'));
        cprintf('blue', 'method:: %s, top_k:: %d, mean rec:: %.2f\n\n', ...
            method, top_k(i), 100*mean_recall);
    end
end