function ret = iter()
%get now iterator of solver
%   Usage: iter()
    ret = caffe_mex('get_solver_iter');
end