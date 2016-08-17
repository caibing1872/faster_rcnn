function imdb = imdb_from_ilsvrc14(root_dir, image_set, flip)
% for train14 dataset (aka, image_set)
%  val/val1/val2
%  train14
%
% imdb.name = 'ilsvrc14_val1'
% imdb.extension = 'JPEG'
% imdb.image_dir = '../ILSVRC2014_DET_train/'
% imdb.details   [structure]
%       image_list_file
%       blacklist_file
%       bbox_path
%       meta_det    [structure]
%       root_dir
%       devkit_path
% imdb.flip  [logic]
% imdb.image_ids = {'ILSVRC2012_val_00010706', ... }
% imdb.sizes = [numimages x 2]
% imdb.is_blacklisted
% imdb.classes = {'accordian', ... }
% imdb.num_classes
% imdb.class_to_id       [Map]
% imdb.class_ids
% imdb.image_at = pointer
% imdb.eval_func = pointer to the function that evaluates detections
% imdb.roidb_func = pointer to the function that returns regions of interest

if ~exist('./imdb/cache/ilsvrc', 'dir')
    mkdir('./imdb/cache/ilsvrc');
end
if flip == false
    cache_file = ['./imdb/cache/ilsvrc/imdb_ilsvrc14_' image_set '_unflip'];
else
    cache_file = ['./imdb/cache/ilsvrc/imdb_ilsvrc14_' image_set '_flip'];
end

try
    load(cache_file);
catch
    NUM_CLS                 = 200;
    bbox_path.train         = fullfile(root_dir, 'ILSVRC2014_DET_bbox_train');
    bbox_path.val           = fullfile(root_dir, 'ILSVRC2013_DET_bbox_val');
    im_path.test            = fullfile(root_dir, 'ILSVRC2013_DET_test');
    im_path.train14         = fullfile(root_dir, 'ILSVRC2014_DET_train');
    im_path.val             = fullfile(root_dir, 'ILSVRC2013_DET_val');
    im_path.val1            = fullfile(root_dir, 'ILSVRC2013_DET_val');
    im_path.val2            = fullfile(root_dir, 'ILSVRC2013_DET_val');
    im_path.val2_no_GT      = fullfile(root_dir, 'ILSVRC2013_DET_val');
    
    im_path.real_test       = fullfile(root_dir, 'ILSVRC2015_DET_test');
    im_path.val1_14         = fullfile(root_dir, 'ILSVRC2014_DET_train');
    im_path.val1_13         = fullfile(root_dir, 'ILSVRC2013_DET_val');
    im_path.pos1k_13        = fullfile(root_dir, 'ILSVRC2014_DET_train');
    
    devkit_path             = fullfile(root_dir, 'ILSVRC2014_devkit');
    
    meta_det                = load(fullfile(devkit_path, 'data', 'meta_det.mat'));
    imdb.name               = ['ilsvrc14_' image_set];
    imdb.extension          = 'JPEG';
    is_blacklisted          = containers.Map;
    
    if strcmp(image_set, 'val') || strcmp(image_set, 'val1') || ...
            strcmp(image_set, 'val2') || strcmp(image_set, 'test') || ...
            strcmp(image_set, 'train14') || strcmp(image_set, 'val2_no_GT') || ...
            strcmp(image_set, 'real_test') || strcmp(image_set, 'val1_13') || ...
            strcmp(image_set, 'val1_14') || strcmp(image_set, 'pos1k_13')
        
        imdb.image_dir = im_path.(image_set);
        imdb.details.image_list_file = ...
            fullfile(devkit_path, 'data', 'det_lists', [image_set '.txt']);
        
        if exist(imdb.details.image_list_file, 'file')
            fid = fopen(imdb.details.image_list_file, 'r');
            temp = textscan(fid, '%s %d');
	    if strcmp(imdb.name, 'ilsvrc14_val1_14') 
		imdb.image_ids = cellfun(@(x) x(1:end-5), temp{1}, 'uniformoutput', false);
            else 
		imdb.image_ids = temp{1}; 
            end   % cell type
        else
            if strcmp(image_set, 'val2_no_GT')
                fid1_temp = fopen(fullfile(devkit_path, 'data', 'det_lists', 'val2_original.txt'), 'r');
                temp = textscan(fid1_temp, '%s %d');
                complete_im_list1 = temp{1};   % cell type    
                fid2_temp = fopen(fullfile(devkit_path, 'data', 'det_lists', 'val2.txt'), 'r');
                temp = textscan(fid2_temp, '%s %d');
                complete_im_list2 = temp{1};       % cell type    
                assert(length(complete_im_list1) >= length(complete_im_list2));
                
                imdb.image_ids = setdiff(complete_im_list1, complete_im_list2);
            else
                error('image_list_file does not exist! %s', imdb.details.image_list_file);
            end
        end       
        imdb.flip = flip;
        
        % blacklist case
        if strcmp(image_set, 'val') || ...
                strcmp(image_set, 'val1') || strcmp(image_set, 'val2')
            
            imdb.details.blacklist_file = ...
                fullfile(devkit_path, 'data', ...
                'ILSVRC2014_det_validation_blacklist.txt');
            [bl_image_ids, ~] = textread(imdb.details.blacklist_file, '%d %s');
            is_blacklisted = containers.Map(bl_image_ids, ones(length(bl_image_ids), 1));
            
        else
            imdb.details.blacklist_file = [];
        end
        % bbox path case
        if strcmp(image_set, 'val') || strcmp(image_set, 'val1') ...
                || strcmp(image_set, 'val2') || strcmp(image_set, 'val2_no_GT')
            imdb.details.bbox_path = bbox_path.val;
        elseif strcmp(image_set, 'train14')
            imdb.details.bbox_path = bbox_path.train;
        end
        
        if flip
            image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
            flip_image_at = @(i) sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
            for i = 1:length(imdb.image_ids)
                if ~exist(flip_image_at(i), 'file')
                    imwrite(fliplr(imread(image_at(i))), flip_image_at(i));
                end
            end
            img_num = length(imdb.image_ids)*2;
            image_ids = imdb.image_ids;
            imdb.image_ids(1:2:img_num) = image_ids;
            imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
            imdb.flip_from = zeros(img_num, 1);
            imdb.flip_from(2:2:img_num) = 1:2:img_num;
        end
        
        % all classes are present in val/test/train14
        imdb.classes = {meta_det.synsets_det(1:NUM_CLS).name};
        imdb.num_classes = length(imdb.classes);
        imdb.class_to_id = containers.Map(imdb.classes, 1:imdb.num_classes);
        imdb.class_ids = 1:imdb.num_classes;
        
        imdb.image_at = @(i) ...
            fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
        
    else
        error('unknown image set');
    end
    
    % private ILSVRC 2014 details
    imdb.details.meta_det    = meta_det;
    imdb.details.root_dir    = root_dir;
    imdb.details.devkit_path = devkit_path;
    % VOC-style specific functions for evaluation and region of interest DB
    imdb.eval_func = @imdb_eval_ilsvrc14;
    imdb.roidb_func = @roidb_from_ilsvrc14;
    
    % read each image to get the 'imdb.sizes'
    % Some images are blacklisted due to noisy annotations
    imdb.is_blacklisted = false(length(imdb.image_ids), 1);
    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
        try
            im = imread(imdb.image_at(i));
        catch lasterror
            if strcmp(lasterror.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
                warning('converting %s from CMYK to RGB', imdb.image_at(i));
                cmd = ['convert ' imdb.image_at(i) ' -colorspace CMYK -colorspace RGB ' imdb.image_at(i)];
                system(cmd);
                im = imread(imdb.image_at(i));
            else
                error(lasterror.message);
            end
        end
        imdb.sizes(i, :) = [size(im, 1) size(im, 2)];
        imdb.is_blacklisted(i) = is_blacklisted.isKey(i);
        
        % faster, but seems to fail on some images :(
        %info = imfinfo(imdb.image_at(i));
        %assert(isscalar(info.Height) && info.Height > 0);
        %assert(isscalar(info.Width) && info.Width > 0);
        %imdb.sizes(i, :) = [info.Height info.Width];
    end
    
    fprintf('Saving imdb to cache...');
    save(cache_file, 'imdb', '-v7.3');
    fprintf('done\n');
end


% ------------------------------------------------------------------------
function wnid = get_wnid(image_id)
% ------------------------------------------------------------------------
ind = strfind(image_id, '_');
wnid = image_id(1:ind-1);
