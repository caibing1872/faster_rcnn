function ret = max_iter()
%get max iterator of solver
%   Usage: max_iter()
    ret = caffe_mex('get_solver_max_iter');
end