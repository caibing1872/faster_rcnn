function roidb = roidb_from_ilsvrc14(imdb,varargin)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParameter('exclude_difficult_samples',       false,   @islogical);
ip.addParameter('with_selective_search',           false,  @islogical);
ip.addParameter('with_edge_box',                   false,  @islogical);
ip.addParameter('with_self_proposal',              false,  @islogical);
ip.addParameter('rootDir',                         '.',    @ischar);
ip.addParameter('extension',                       '',     @ischar);
ip.addParameter('roidb_name_suffix',               '',     @isstr);
ip.parse(imdb, varargin{:});
opts = ip.Results;

if ~exist('./imdb/cache/ilsvrc', 'dir')
    mkdir('./imdb/cache/ilsvrc');
end

try
    flip = imdb.flip;
catch
    flip = false;
end

if ~isempty(opts.roidb_name_suffix)
    assert(strcmp(imdb.name, 'ilsvrc14_val2'));
    assert(flip == false); 
end

if flip == false
    cache_file = ['./imdb/cache/ilsvrc/roidb_' imdb.name '_unflip'];
    if ~isempty(opts.roidb_name_suffix)
        cache_file = [cache_file '_' opts.roidb_name_suffix '.mat'];
        try
            %disp(cache_file); 
            load(cache_file);
            roidb.rois = rois;
            roidb.name = imdb.name;
        catch
            error('fuck you!!!');
        end
        return;
    end
else
    cache_file = ['./imdb/cache/ilsvrc/roidb_' imdb.name '_flip'];
end

%%
try
    load(cache_file);
catch
    addpath(fullfile(imdb.details.devkit_path, 'evaluation'));
    
    roidb.name = imdb.name;
    % wsh  regions_file = fullfile('data', 'selective_search_data', [roidb.name '.mat']);    
    regions = [];
    if opts.with_selective_search
        fprintf('Loading SS region proposals...');
        regions = load_proposals(regions_file_ss, regions);
        fprintf('done\n');
    end
    if opts.with_edge_box
        regions = load_proposals(regions_file_eb, regions);
    end
    if opts.with_self_proposal
        regions = load_proposals(regions_file_sp, regions);
    end
    
    if isempty(regions)
        fprintf('Warrning: no ADDITIONAL windows proposal is loaded!\n');
        regions.boxes = cell(length(imdb.image_ids), 1);
        if flip
            regions.images = imdb.image_ids(1:2:end);
        else
            regions.images = imdb.image_ids;
        end
    end
    
    hash = make_hash(imdb.details.meta_det.synsets_det);
    if ~flip
        
        for i = 1:length(imdb.image_ids)
            
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
            anno_file = fullfile(imdb.details.bbox_path, [imdb.image_ids{i} '.xml']);
            
            try
                rec = VOCreadrecxml(anno_file, hash);
            catch
                error('GT(xml) file empty/broken: %s\n', imdb.image_ids{i});
            end
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            roidb.rois(i) = attach_proposals(rec, regions.boxes{i}, imdb.class_to_id, opts.exclude_difficult_samples, false);
        end
    else
        % flip case
        for i = 1:length(imdb.image_ids)/2
            
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
            anno_file = fullfile(imdb.details.bbox_path, [imdb.image_ids{2*i-1} '.xml']);
            
            try
                rec = VOCreadrecxml(anno_file, hash);
            catch
                error('GT(xml) file empty/broken: %s\n', imdb.image_ids{2*i-1});
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
% per IMAGE
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

    gt_classes = cat(1, ilsvrc_rec.objects(:).label);
    num_gt_boxes = size(gt_boxes, 1);
else
    gt_boxes = [];
    all_boxes = boxes;
    gt_classes = [];
    num_gt_boxes = 0;
end

num_boxes = size(boxes, 1);
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
% 'overlap' is (4 gt + 3 provided_box) x 200 classes
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
    rec.overlap(:, gt_classes(i)) = ...
        max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end


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
