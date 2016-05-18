function ret = train( input_cell, step)
%train several step with same data
%   Usage: train(batch_data, 10) or train(batch_data)
    if nargin < 2
        step = 1;
    end
    assert(iscell(input_cell), 'Input should be cell.');
    while step > 0
        step = step - 1;
        ret = caffe_mex('train', input_cell);
    end
end

