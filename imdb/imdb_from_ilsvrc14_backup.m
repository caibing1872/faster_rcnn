function imdb = imdb_from_ilsvrc14(root_dir, image_set, flip)
%
% for each of the 200 classes there's a
%  train_pos_X
%  train_neg_X
%  train_part_X (multiple objects, hard cases)
%
% and there's also
%  val/val1/val2
%  test
%  split val into two folds with roughly equal # instances per class

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/ILSVRC/ILSVRC2013_DET_train/n02672831/'
%imdb.extension = 'JPEG'
%imdb.details   [structure]
%       image_list_file
%       blacklist_file
%       bbox_path
%       meta_det    [structure]
%       root_dir
%       devkit_path
%imdb.flip  [logic]
%imdb.image_ids = {'n02672831_11478', ... }
%imdb.sizes = [numimages x 2]
%imdb.is_blacklisted
%imdb.classes = {'accordian', ... }
%imdb.num_classes
%imdb.class_to_id       [Map]
%imdb.class_ids
%imdb.image_at = pointer
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

if ~exist('./imdb/cache/ilsvrc', 'dir')
    mkdir('./imdb/cache/ilsvrc');
end
cache_file = ['./imdb/cache/ilsvrc/imdb_ilsvrc14_' image_set];
try
    load(cache_file);
catch
    NUM_CLS         = 200;
    bbox_path.train = fullfile(root_dir, 'ILSVRC2014_DET_bbox_train');
    bbox_path.val   = fullfile(root_dir, 'ILSVRC2013_DET_bbox_val');
    im_path.test    = fullfile(root_dir, 'ILSVRC2013_DET_test');
    im_path.train   = fullfile(root_dir, 'ILSVRC2014_DET_train');
    % unchanged in 2014
    im_path.val     = fullfile(root_dir, 'ILSVRC2013_DET_val');
    im_path.val1    = fullfile(root_dir, 'ILSVRC2013_DET_val');
    im_path.val2    = fullfile(root_dir, 'ILSVRC2013_DET_val');   
    devkit_path     = fullfile(root_dir, 'ILSVRC2014_devkit');
    
    meta_det        = load(fullfile(devkit_path, 'data', 'meta_det.mat'));
    imdb.name       = ['ilsvrc14_' image_set];
    imdb.extension  = 'JPEG';
    is_blacklisted  = containers.Map;
    
    % derive image directory
    match = regexp(image_set, 'train_pos_(?<class_num>\d+)', 'names');
    if ~isempty(match)
        
        class_num = str2double(match.class_num);
        assert(class_num >= 1 && class_num <= NUM_CLS);
        imdb.image_dir = im_path.train;
        imdb.details.image_list_file = ...
            fullfile(devkit_path, 'data', 'det_lists', [image_set '.txt']);
        fid = fopen(imdb.details.image_list_file, 'r');
        temp = textscan(fid, '%s');
        imdb.image_ids = temp{1};   % cell type
        
        % only one class is present in train_pos
        imdb.classes = {meta_det.synsets_det(class_num).name};
        imdb.num_classes = length(imdb.classes);
        imdb.class_to_id = ...
            containers.Map(imdb.classes, class_num);
        imdb.class_ids = class_num;
        
%         imdb.image_at = @(i) ...
%             fullfile(imdb.image_dir, get_wnid(imdb.image_ids{i}), ...
%             [imdb.image_ids{i} '.' imdb.extension]);       
        imdb.image_at = @(i) ...
            fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
        
        imdb.details.blacklist_file = [];
        imdb.details.bbox_path = bbox_path.train;
        
    elseif strcmp(image_set, 'val') || strcmp(image_set, 'val1') || ...
            strcmp(image_set, 'val2') || strcmp(image_set, 'test')
        
        imdb.image_dir = im_path.(image_set);
        imdb.details.image_list_file = ...
            fullfile(devkit_path, 'data', 'det_lists', [image_set '.txt']);
        fid = fopen(imdb.details.image_list_file, 'r');
        temp = textscan(fid, '%s %d');
        imdb.image_ids = temp{1};   % cell type
                
        imdb.extension = 'JPEG';
        imdb.flip = flip;
        
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
        
        % all classes are present in val/test
        imdb.classes = {meta_det.synsets_det(1:NUM_CLS).name};
        imdb.num_classes = length(imdb.classes);
        imdb.class_to_id = ...
            containers.Map(imdb.classes, 1:imdb.num_classes);
        imdb.class_ids = 1:imdb.num_classes;
        
        imdb.image_at = @(i) ...
            fullfile(imdb.image_dir, [imdb.image_ids{i} '.' imdb.extension]);
              
        if ~strcmp(image_set, 'test')
            imdb.details.bbox_path = bbox_path.val;
        end
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
    
    % Some images are blacklisted due to noisy annotations
    imdb.is_blacklisted = false(length(imdb.image_ids), 1);
    
    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
        try
            im = imread(imdb.image_at(i));
        catch lasterror  
            % hah, annoying data issues
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
