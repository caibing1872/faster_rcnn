function set_gpus( gpu_ids )
%Set gpu-ids using in after works
%   Usage: set_gpus([0 1 2 3])
    caffe_mex('set_device_solver', gpu_ids);
end

