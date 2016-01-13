function [anchors, output_width_map, output_height_map] = ...
    proposal_prepare_anchors(conf, cache_name, test_net_def_file)

    [output_width_map, output_height_map] = ...
        proposal_calc_output_size(conf, test_net_def_file);
    
    anchors = proposal_generate_anchors(cache_name, 'scales',  2.^[3:5]);
end