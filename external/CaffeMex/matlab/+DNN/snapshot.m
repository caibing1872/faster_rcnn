function snapshot(filenameStr)
%train several step with same data
%   Usage: snapshot(filenameStr)
    caffe_mex('snapshot', filenameStr);
end

