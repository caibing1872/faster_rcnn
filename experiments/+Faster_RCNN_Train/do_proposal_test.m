function roidb_new = do_proposal_test(conf, model_stage, imdb, roidb)

% revised by hyli
cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', model_stage.cache_name, imdb.name);
try
    ld = load(fullfile(cache_dir, ['aboxes_filtered_' imdb.name '.mat']));
    aboxes = ld.aboxes;
    clear ld;
catch
    % save 'aboxes' in the cache, like:
    %   rpn_cachedir/ilsvrc13_vgg_stage1_rpn/ilsvrc13_val1/proposal_boxes_ilsvrc13_val1.mat
    aboxes = proposal_test(conf, imdb, ...
        'net_def_file',     model_stage.test_net_def_file, ...
        'net_file',         model_stage.output_model_file, ...
        'cache_name',       model_stage.cache_name);
    
    % the following is extremely time-consuming
    aboxes = boxes_filter(aboxes, model_stage.nms.per_nms_topN, ...
        model_stage.nms.nms_overlap_thres, model_stage.nms.after_nms_topN, conf.use_gpu);
    save(fullfile(cache_dir, ['aboxes_filtered_' imdb.name '.mat']), 'aboxes', '-v7.3');
end

roidb_regions = make_roidb_regions(aboxes, imdb.image_ids);

% update: change some code to save memory
try
    ld = load(fullfile(cache_dir, 'trick_new_roidb.mat'));
    rois = ld.rois;
    assert(length(rois) == length(roidb.rois));
    clear ld;
catch
    fprintf('update roidb/rois during test, again, taking quite a while (brew some coffe or take a walk!:)...\n');
    
    roidb_from_proposal(imdb, roidb, roidb_regions, 'keep_raw_proposal', false, 'mat_file_prefix', cache_dir);
    
    ld = load(fullfile(cache_dir, 'trick_new_roidb.mat'));
    rois = ld.rois;
    assert(length(rois) == length(roidb.rois));
    clear ld;
end
roidb_new.rois = rois;

end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)

% to speed up nms
if per_nms_topN > 0
    aboxes = cellfun(@(x) x(1:min(length(x), per_nms_topN), :), aboxes, 'UniformOutput', false);
end
% do nms
fprintf('do nms during test, taking quite a while (brew some coffe or take a walk!:)...\n');
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    if use_gpu
        for i = 1:length(aboxes)
            aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres, use_gpu), :);
        end
    else
        parfor i = 1:length(aboxes)
            aboxes{i} = aboxes{i}(nms(aboxes{i}, nms_overlap_thres), :);
        end
    end
end
aver_boxes_num = mean(cellfun(@(x) size(x, 1), aboxes, 'UniformOutput', true));
fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
if after_nms_topN > 0
    aboxes = cellfun(@(x) x(1:min(length(x), after_nms_topN), :), aboxes, 'UniformOutput', false);
end
end

function regions = make_roidb_regions(aboxes, images)
regions.boxes = aboxes;
regions.images = images;
end
