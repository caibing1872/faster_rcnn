function set_weight(weights)
%Set all weights of net
%   Usage: set_weight()
    assert(isstruct(weights));
    caffe_mex('set_weights_solver', weights);
end