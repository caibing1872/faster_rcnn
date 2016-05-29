function roidb = roidb_from_ilsvrc14(imdb,varargin)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

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

if ~exist('./imdb/cache/ilsvrc', 'dir')
    mkdir('./imdb/cache/ilsvrc');
end
cache_file = ['./imdb/cache/ilsvrc/roidb_' imdb.name];
%%
try
    load(cache_file);
catch
    addpath(fullfile(imdb.details.devkit_path, 'evaluation'));
    
    roidb.name = imdb.name;
    try
        flip = imdb.flip;
    catch
        flip = false;
    end
    
%     is_train = false;  % just used for finding the GT files, not real sets of training
%     match = regexp(roidb.name, 'ilsvrc14_train_pos_(?<class_num>\d+)', 'names');
%     if ~isempty(match)
%         is_train = true;
%     end
    
    % wsh  regions_file = fullfile('data', 'selective_search_data', [roidb.name '.mat']);
    fprintf('Loading region proposals...');
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
        if flip
            regions.images = imdb.image_ids(1:2:end);
        else
            regions.images = imdb.image_ids;
        end
    end
    
    hash = make_hash(imdb.details.meta_det.synsets_det);
    if ~flip
        % non-flip case: train_pos must fall into this case;
        %                   val1/val2/val/test maybe have this attribute
        
        for i = 1:length(imdb.image_ids)
            
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
%             if is_train
%                 anno_file = fullfile(imdb.details.bbox_path, ...
%                     get_wnid(imdb.image_ids{i}), [imdb.image_ids{i} '.xml']);
%             else
            anno_file = fullfile(imdb.details.bbox_path, [imdb.image_ids{i} '.xml']);
%             end
            
            try
                rec = VOCreadrecxml(anno_file, hash);
            catch
                rec = [];
                warning('GT(xml) file empty: %s\n', imdb.image_ids{i});
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i) = attach_proposals(rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, false);
        end
    else
        % flip case: val1/val2/val/test maybe have this attribute
        for i = 1:length(imdb.image_ids)/2
            
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
            anno_file = fullfile(imdb.details.bbox_path, [imdb.image_ids{2*i-1} '.xml']);
            
            try
                rec = VOCreadrecxml(anno_file, hash);
            catch
                rec = []; 
                warning('GT(xml) file empty: %s\n', imdb.image_ids{2*i-1});
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
if isfield(ilsvrc_rec, 'objects') && ~isempty(ilsvrc_rec.objects)
    if exclude_difficult_samples
        valid_objects = ~cat(1, ilsvrc_rec.objects(:).difficult);
    else
        valid_objects = 1:length(ilsvrc_rec.objects(:));
    end    
    gt_boxes = cat(1, ilsvrc_rec.objects(valid_objects).bbox);
    %%% ============ NOTE ==============
    % coordinate starts from 0 in ilsvrc
    gt_boxes = gt_boxes + 1;
    
    if flip
        gt_boxes(:, [1, 3]) = ilsvrc_rec.imgsize(1) + 1 - gt_boxes(:, [3, 1]);
    end
    all_boxes = cat(1, gt_boxes, boxes);

    %gt_classes = class_to_id.values({ilsvrc_rec.objects(valid_objects).class});
    %gt_classes = cat(1, gt_classes{:});

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

% added by hyli to solve a bug when handling train_pos
if class_to_id.Count == 1
    temp_cnt = 200;
else
    temp_cnt = class_to_id.Count;
end
rec.overlap = zeros(num_gt_boxes+num_boxes, temp_cnt, 'single');
for i = 1:num_gt_boxes
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions)
% ------------------------------------------------------------------------
if isempty(regions)
    regions = load(proposal_file);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes(:), regions_more.boxes(:), 'UniformOutput', false);
end

% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
