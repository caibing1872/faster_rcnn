function dataset = ilsvrc14(dataset, usage, use_flip, root_path, roidb_name_suffix)
% refactored by hyli
%
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_test

if nargin <= 4, roidb_name_suffix = ''; end

switch usage
    
%     case {'train'}
%         % DEPRECATED
%         % we use all the training pos and val1 (defined by the genius Ross)
%         dataset.imdb_train{1,1} = imdb_from_ilsvrc14(root_path, 'val1', use_flip);
%         
%         for i = 1:200
%             dataset.imdb_train{i+1,1} = imdb_from_ilsvrc14(root_path, ...
%                 sprintf('train_pos_%d', i));
%         end
%         dataset.roidb_train = cellfun(@(x) x.roidb_func(x), ...
%             dataset.imdb_train, 'UniformOutput', false);
        
    case {'train14'}
        % IN SERVICE
        % we use all the 2014 training pos and val1 
        dataset.imdb_train{1,1} = imdb_from_ilsvrc14(root_path, 'val1', use_flip);
        dataset.imdb_train{2,1} = imdb_from_ilsvrc14(root_path, 'train14', use_flip);
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x), ...
            dataset.imdb_train, 'UniformOutput', false);
        
    case {'train_val1'}
        % IN SERVICE
        dataset.imdb_train{1,1} = imdb_from_ilsvrc14(root_path, 'val1', use_flip);
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
        
    case {'test'}
        % IN SERVICE
        dataset.imdb_test     = imdb_from_ilsvrc14(root_path, 'val2', false);
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, ...
            'roidb_name_suffix', roidb_name_suffix);
        
    case {'test_no_gt'}
        dataset.imdb_test     = imdb_from_ilsvrc14(root_path, 'val2_no_GT', false);
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
        
    otherwise
        error('usage = ''train14'' or ''test''');
end

end
