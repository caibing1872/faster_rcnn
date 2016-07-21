function roidb_from_proposal(imdb, roidb, regions, varargin)
% roidb = roidb_from_proposal(imdb, roidb, regions, varargin)s
% --------------------------------------------------------
% Note:
%       this file has been revised by save memory
% --------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb',                      @isstruct);
ip.addRequired('roidb',                     @isstruct);
ip.addRequired('regions',                   @isstruct);
ip.addParameter('keep_raw_proposal', true,  @islogical);
ip.addParameter('mat_file', '',      @isstr);

ip.parse(imdb, roidb, regions, varargin{:});
opts = ip.Results;

assert(strcmp(opts.roidb.name, opts.imdb.name));
rois = opts.roidb.rois;

if ~opts.keep_raw_proposal
    % remove proposal boxes in roidb
    for i = 1:length(rois)
        is_gt = rois(i).gt;
        rois(i).gt = rois(i).gt(is_gt, :);
        rois(i).overlap = rois(i).overlap(is_gt, :);
        rois(i).boxes = rois(i).boxes(is_gt, :);
        rois(i).class = rois(i).class(is_gt, :);
    end
end

% chunk = 2000 takes 20G ROM
chunk = 500;
%if imdb.flip, FLIP = 'flip'; else FLIP = 'unflip'; end
%mat_name = fullfile(opts.mat_file_prefix, ['roidb_' roidb.name '_' FLIP '_1.mat']);
mat_name = opts.mat_file;
m = matfile(mat_name, 'Writable', true);

image_ids_ = imdb.image_ids;
% newly-generated
images_ = opts.regions.images;
boxes_ = opts.regions.boxes;

% add new proposal boxes
for i = 1 : ceil(length(rois)/chunk)
    
    start_ind = 1 + chunk*(i-1);
    end_ind = min(length(rois), chunk + chunk*(i-1));
    sub_length = end_ind - start_ind + 1;
    
    % init empty structure called temp
    temp = struct('gt', [], 'overlap', [], 'boxes', [], 'class', []);
    temp(sub_length).gt = [];
    
    parfor kk = 1:sub_length
        % 'kk' is relative index
        abs_ind = kk + (i-1)*chunk;
        [~, image_name1] = fileparts(image_ids_{abs_ind});
        [~, image_name2] = fileparts(images_{abs_ind});
        assert(strcmp(image_name1, image_name2));
        
        boxes = boxes_{abs_ind}(:, 1:4)      
        gt_boxes = rois(abs_ind).boxes(rois(abs_ind).gt, :);
        gt_classes = rois(abs_ind).class(rois(abs_ind).gt, :);
        all_boxes = cat(1, rois(abs_ind).boxes, boxes);
        
        num_gt_boxes = size(gt_boxes, 1);
        num_boxes = size(boxes, 1);
        
        temp(kk).gt = cat(1, rois(abs_ind).gt, false(num_boxes, 1));
        temp(kk).overlap = cat(1, rois(abs_ind).overlap, zeros(num_boxes, size(rois(abs_ind).overlap, 2)));
        temp(kk).boxes = cat(1, rois(abs_ind).boxes, boxes);
        temp(kk).class = cat(1, rois(abs_ind).class, zeros(num_boxes, 1));
                
        for j = 1 : num_gt_boxes
            temp(kk).overlap(:, gt_classes(j)) = ...
                max( full( temp(kk).overlap(:, gt_classes(j)) ), boxoverlap(all_boxes, gt_boxes(j, :)) );
        end
        temp(kk).overlap = sparse(double( temp(kk).overlap ));
    end
    m.rois(1, start_ind : end_ind) = temp;
end

% ========= the original code =========
% =====================================
% % add new proposal boxes
% for i = 1:length(rois)
%     [~, image_name1] = fileparts(imdb.image_ids{i});
%     [~, image_name2] = fileparts(opts.regions.images{i});
%     assert(strcmp(image_name1, image_name2));
% 
%     boxes = opts.regions.boxes{i}(:, 1:4);
%     is_gt = rois(i).gt;
%     gt_boxes = rois(i).boxes(is_gt, :);
%     gt_classes = rois(i).class(is_gt, :);
%     all_boxes = cat(1, rois(i).boxes, boxes);
% 
%     num_gt_boxes = size(gt_boxes, 1);
%     num_boxes = size(boxes, 1);
% 
%     rois(i).gt = cat(1, rois(i).gt, false(num_boxes, 1));
%     rois(i).overlap = cat(1, rois(i).overlap, zeros(num_boxes, size(rois(i).overlap, 2)));
%     rois(i).boxes = cat(1, rois(i).boxes, boxes);
%     rois(i).class = cat(1, rois(i).class, zeros(num_boxes, 1));
%     for j = 1:num_gt_boxes
%         rois(i).overlap(:, gt_classes(j)) = ...
%             max(full(rois(i).overlap(:, gt_classes(j))), boxoverlap(all_boxes, gt_boxes(j, :)));
%     end
%     % this is inspired by shwang
%     rois(i).overlap = sparse(double(rois(i).overlap));
% end
% roidb.rois = rois;

end