function recovery( solver_file, solverstate_file )
%Init a caffe multi-gpu solver
%   Usage: recovery(solver_file, solverstate_file)
%   Before everytime you call this function, 
%   you need to call set_gpus() first.
    assert(strfind(solverstate_file, '.solverstate')>0, 'recovery() only accept solverstate file.');
    caffe_mex('recovery_solver', solver_file, solverstate_file);
end

