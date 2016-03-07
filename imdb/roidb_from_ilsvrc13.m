function roidb = roidb_from_ilsvrc13(imdb,varargin)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('exclude_difficult_samples',       false,   @islogical);
ip.addParamValue('with_selective_search',           false,  @islogical);
ip.addParamValue('with_edge_box',                   false,  @islogical);
ip.addParamValue('with_self_proposal',              false,  @islogical);
ip.addParamValue('rootDir',                         '.',    @ischar);
ip.addParamValue('extension',                       '',     @ischar);
ip.parse(imdb, varargin{:});
opts = ip.Results;
%%
cache_file = ['./imdb/cache/roidb_' imdb.name];
%%
try
    load(cache_file);
catch
    addpath(fullfile(imdb.details.devkit_path, 'evaluation'));
    
    roidb.name = imdb.name;
    
    is_train = false;
    match = regexp(roidb.name, 'ilsvrc13_train_pos_(?<class_num>\d+)', 'names');
    if ~isempty(match)
        is_train = true;
    end
    is_test = false;
    if strcmp(roidb.name, 'ilsvrc13_test');
        is_test = true;
    end
    
    % wsh  regions_file = fullfile('data', 'selective_search_data', [roidb.name '.mat']);
    regions = [];
    if opts.with_selective_search
        regions = load_proposals(regions_file_ss, regions);
    end
    if opts.with_edge_box
        regions = load_proposals(regions_file_eb, regions);
    end
    if opts.with_self_proposal
        regions = load_proposals(regions_file_sp, regions);
    end
        
    fprintf('done\n');
    
  if isempty(regions)
      fprintf('Warrning: no windows proposal is loaded !\n');
      regions.boxes = cell(length(imdb.image_ids), 1);
      if imdb.flip
            regions.images = imdb.image_ids(1:2:end);
      else
            regions.images = imdb.image_ids;
      end
  end
  
    hash = make_hash(imdb.details.meta_det.synsets);
    imdb.flip
    if ~imdb.flip
        for i = 1:length(imdb.image_ids)
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
            if is_train
                anno_file = fullfile(imdb.details.bbox_path, ...
                    get_wnid(imdb.image_ids{i}), [imdb.image_ids{i} '.xml']);
            elseif is_test
                anno_file = [];
            else
                anno_file = fullfile(imdb.details.bbox_path, ...
                    [imdb.image_ids{i} '.xml']);
            end
            
            try
                rec = VOCreadrecxml(anno_file, hash);
            catch
                rec = [];
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i) = attach_proposals(rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, false);
        end
    else
        for i = 1:length(imdb.image_ids)/2
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
            if is_train
                anno_file = fullfile(imdb.details.bbox_path, ...
                    get_wnid(imdb.image_ids{2*i-1}), [imdb.image_ids{2*i-1} '.xml']);
            elseif is_test
                anno_file = [];
            else
                anno_file = fullfile(imdb.details.bbox_path, ...
                    [imdb.image_ids{2*i-1} '.xml']);
            end
            
            try
                rec = VOCreadrecxml(anno_file, hash);
            catch
                rec = []; %exclude_difficult_samples
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
                assert(imdb.flip_from(i*2) == i*2-1);
            end
            roidb.rois(i*2-1) = attach_proposals(rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, false);
            roidb.rois(i*2) = attach_proposals(rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, true);
        end
    end
    
    
    
    rmpath(fullfile(imdb.details.devkit_path, 'evaluation'));
    
    fprintf('Saving roidb to cache...');
    save(cache_file, 'roidb', '-v7.3');
    fprintf('done\n');
end


% ------------------------------------------------------------------------
function rec = attach_proposals(ilsvrc_rec, boxes, class_to_id, exclude_difficult_samples, flip)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
if ~isempty(boxes)
    boxes = boxes(:, [2 1 4 3]);
    if flip
        boxes(:, [1, 3]) = ilsvrc_rec.imgsize(1) + 1 - boxes(:, [3, 1]);
    end
end

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(ilsvrc_rec, 'objects') && length(ilsvrc_rec.objects) > 0
    if exclude_difficult_samples
        valid_objects = ~cat(1, ilsvrc_rec.objects(:).difficult);
    else
        valid_objects = 1:length(ilsvrc_rec.objects(:));
    end
    gt_boxes = cat(1, ilsvrc_rec.objects(valid_objects).bbox);
    if flip
        gt_boxes(:, [1, 3]) = ilsvrc_rec.imgsize(1) + 1 - gt_boxes(:, [3, 1]);
    end
    all_boxes = cat(1, gt_boxes, boxes);
    %
    %gt_classes = class_to_id.values({ilsvrc_rec.objects(valid_objects).class});
    %gt_classes = cat(1, gt_classes{:});
    %
    gt_classes = cat(1, ilsvrc_rec.objects(:).label);
    num_gt_boxes = size(gt_boxes, 1);
else
    gt_boxes = [];
    all_boxes = boxes;
    gt_classes = [];
    num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
