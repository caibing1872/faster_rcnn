function init( solver_file, caffemodel_file )
%Init a caffe multi-gpu solver
%   Usage: init(solver_file, caffemodel_file)
%   Before everytime you call this function, 
%   you need to call set_gpus() first.
    caffe_mex('init_solver', solver_file, caffemodel_file);
end

