
%%1.At each time you use CaffeMex, you need clear mex first
clear mex;

%%2.Example for NET
%set gpu-ids you use
DNN.caffe_mex('set_device', 0:3);
%init a NET with random weights
DNN.caffe_mex('init', 'net.prototxt', '');
%OR init a NET with pre-trained weights
DNN.caffe_mex('init', 'net.prototxt', 'net.caffemodel');

%prepare bottom data (I use random data to give an example, and we do not need data layer any more)
data{1} = single(rand(224,224,3,batch_size_each_card));
data{2} = single(zeros(1,1,1,batch_size_each_card));
%forward 5 times
for i = 1:5
    DNN.caffe_mex('forward', data);
    disp(i);
end
%set diff(DO NOT use it unless you need to implement loss layer in matlab by yourself)
top_diff{1} = single(ones(1,1,1,10));
DNN.caffe_mex('backward', top_diff);
%get all weights of net
weight = DNN.caffe_mex('get_weights');
%get a feature-map
response = DNN.caffe_mex('get_response', 'pool5');
%release NET's memory and shutdown glog
DNN.caffe_mex('release');

%%3.Example for SOLVER
%init log to a specific location
DNN.caffe_mex('init_log', './log/solver_log')
%set multi-gpu ids
DNN.caffe_mex('set_device_solver', 0:0);
%init a solver with prototxt and caffemodel(can be empty)
DNN.caffe_mex('init_solver', 'solver.prototxt', '', 'log/');

%prepare input data for multi_gpu training
clear data;
for i = 1 : gpu_nums
    data{i}{1} = single(img);				%img data
    data{i}{2} = single(zeros(1,1,1,1)); 	%label
end

for i = 1:5 %train 5 iters
    ret = DNN.caffe_mex('train', data);
    fprintf('Loss=%f, acc=%f\n', ret(1).results, ret(2).results)
end
%release SOLVER's memory and shutdown glog
DNN.caffe_mex('release_solver');
