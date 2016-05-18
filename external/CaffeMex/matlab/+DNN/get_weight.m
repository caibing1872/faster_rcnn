function ret = get_weight()
%Get all weights of net
%   Usage: get_weight()
    ret = caffe_mex('get_weights_solver');
end