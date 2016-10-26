% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from ilsvrc 2015/2014, the standard officical version.

%clc;
clear;
close all;
run('startup');
%%
which_method = 'ss'; %'edgebox';
total_chunk = 8;
curr_chunk = 1;
which_set = 'train'; % 'val';
top_k = 1000;

%%
result_name = sprintf('hyli_%s_default_settting', which_method);
if strcmp(which_method, 'ss')
    addpath(genpath('./functions/external_prop/selective_search'));
    
elseif strcmp(which_method, 'edgebox')
    addpath(genpath('./functions/external_prop/edgebox'));
    [model, opts] = hyli_edge_init();
end

save_path = ['./box_proposals/external/' result_name];
mkdir_if_missing(save_path);

ld = load('../cvpr17_proposal/data/imdb/train_val_list.mat');
if strcmp(which_set, 'train');
    full_im_list = ld.train_list;
elseif strcmp(which_set, 'val');
    full_im_list = ld.val_list;
end
full_num_images = length(full_im_list);
ck_interval = ceil(full_num_images/total_chunk);
start_ind = 1 + (curr_chunk-1)*ck_interval;
end_ind = min(ck_interval + (curr_chunk-1)*ck_interval, full_num_images);
num_images = end_ind - start_ind + 1;
im_list = full_im_list(start_ind : end_ind);  % part of the whole set

root_folder{1} = '../cvpr17_proposal/data/datasets/ilsvrc14_det/ILSVRC2013_DET_val';
root_folder{2} = '../cvpr17_proposal/data/datasets/ilsvrc14_det/ILSVRC2014_DET_train';

box_result(num_images).name = '';
box_result(num_images).box = [];
save_mat_file = [save_path sprintf('/%s_ck%d_absInd_%d_%d_total%d.mat', ...
    which_set, curr_chunk, start_ind, end_ind, full_num_images)];

%%
t = tic;
for i = 1 : num_images
    
    name_temp = im_list{i}(21:end);
    box_result(i).name = name_temp;
    if name_temp(1) == 't'
        im_path = fullfile(root_folder{2}, name_temp(7:end));
    elseif name_temp(1) == 'v'
        im_path = fullfile(root_folder{1}, name_temp(5:end));
    end
    
    try
        image = imread(im_path);
    catch lasterror
        % hah, annoying data issues
        if strcmp(lasterror.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
            warning('converting %s from CMYK to RGB', im_path);
            cmd = ['convert ' im_path ' -colorspace CMYK -colorspace RGB ' im_path];
            system(cmd);
            image = imread(im_path);
        else
            error(lasterror.message);
        end
    end
    if size(image, 3) == 1, image = repmat(image, [1 1 3]); end
    
    if strcmp(which_method, 'edgebox')
        
        boxes = edgeBoxes(image, model, opts);
        % [x, y, w, h, score]
        boxes(:, 3) = boxes(:, 1) + boxes(:, 3) - 1;
        boxes(:, 4) = boxes(:, 2) + boxes(:, 4) - 1;
        
    elseif strcmp(which_method, 'ss')
        
        try
            [temp, score] = selective_search_boxes(image);
            % [y1 x1 y2 x2]
            temp = temp(:, [2 1 4 3]);
            boxes = [temp, score];
        catch
            % in some fucking case, ss fails!
            boxes = [1 1 size(image, 2), size(image, 1) 1];
        end
    end
    [~, ind] = sort(boxes(:,5), 'descend');
    boxes = boxes(ind, :);
    box_result(i).box = single(boxes(1:min(top_k, size(boxes,1)), 1:4));
    
    if mod(i, 10) == 1 || i == num_images
        take = toc(t)/(3600*10);
        time_left = take*(num_images-i);
        fprintf('%s, ck# %d (%d-%d), progress i/total, %d/%d, %.3f hrs left ...\n', ...
            which_set, curr_chunk, start_ind, end_ind, i, num_images, time_left);
        t = tic;
    end
end
save(save_mat_file, 'box_result', '-v7.3');
