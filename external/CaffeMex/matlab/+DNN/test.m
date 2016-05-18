function ret = test( input_cell )
%train several step with same data
%   Usage: train(batch_data, 10) or train(batch_data)
    assert(iscell(input_cell), 'Input should be cell.');
    ret = caffe_mex('test', input_cell);
end

