function ret = response(layerName)
%Get feature-map of a specific layer
%   Usage: response(layerName)
    ret = caffe_mex('get_response_solver', layerName);
end