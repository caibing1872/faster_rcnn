function [ model, opts ] = hyli_edge_init()

    model = load('./functions/external_prop/edgebox/models/forest/modelBsds');
    model = model.model;
    model.opts.multiscale=0;
    model.opts.sharpen=2;
    model.opts.nThreads=4;
    % set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .65;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e4;  % max number of boxes to detect


end

