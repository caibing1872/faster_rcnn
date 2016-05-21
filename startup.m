function startup()
% startup()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

% note by hyli: refactor branch

% Shall fix error: An unexpected error occurred during CUDA execution. The
% CUDA error was: cannot set while device is active in this process.
%     a = gpuArray(1); 
%     clear a;
    curdir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(curdir, 'utils')));
    addpath(genpath(fullfile(curdir, 'functions')));
    addpath(genpath(fullfile(curdir, 'bin')));
    addpath(genpath(fullfile(curdir, 'experiments')));
    addpath(genpath(fullfile(curdir, 'imdb')));

    mkdir_if_missing(fullfile(curdir, 'datasets'));

    mkdir_if_missing(fullfile(curdir, 'external'));

    caffe_path = fullfile(curdir, 'external', 'CaffeMex', 'matlab');
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));

    mkdir_if_missing(fullfile(curdir, 'imdb', 'cache'));

    mkdir_if_missing(fullfile(curdir, 'output'));

    mkdir_if_missing(fullfile(curdir, 'models'));

    fprintf('fast_rcnn startup done\n');
end
