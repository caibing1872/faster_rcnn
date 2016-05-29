function dataset = ilsvrc14(dataset, usage, use_flip, root_path)
% refactored by hyli
%
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_test

switch usage
    
    case {'train'}
        % we use all the training pos and val1 (defined by the genius Ross)
        dataset.imdb_train{1,1} = imdb_from_ilsvrc14(root_path, 'val1', use_flip);
        
        for i = 1:200
            dataset.imdb_train{i+1,1} = imdb_from_ilsvrc14(root_path, ...
                sprintf('train_pos_%d', i));
        end
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x), ...
            dataset.imdb_train, 'UniformOutput', false);
        
    case {'train_val1'}
        
        dataset.imdb_train{1,1} = imdb_from_ilsvrc14(root_path, 'val1', use_flip);
        dataset.roidb_train = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
        
    case {'test'}
        dataset.imdb_test     = imdb_from_ilsvrc14(root_path, 'val2', false);
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''train'' or ''test''');
end

end
