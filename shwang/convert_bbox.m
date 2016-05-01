function abs_bbox = convert_bbox(relative_bbox, im, fixed_length)
% convert the relative bbox value to absolute value based on the size of
% image; if fixed_length is chosen, then resize to a square result.
%
% input:
%   relative_bbox:      num x 4
%                       where each entry ranges from 0 to 1 relative to
%                       original image.


% TEMPORAL:
if numel(im) == 2
    % treat 'im' as [h w]
    h = double(im(1));
    w = double(im(2));
else
    % treat 'im' as image
    if nargin == 3
        w = fixed_length;
        h = fixed_length;
    else
        [h, w, ~] = size(im);
    end
end

abs_bbox = zeros(size(relative_bbox, 1), 4);

for i = 1:size(relative_bbox, 1)
    
    abs_bbox(i, :) = [ relative_bbox(i, 1)*w relative_bbox(i, 2)*h ...
        relative_bbox(i, 3)*w relative_bbox(i, 4)*h ...
        ];
end

abs_bbox = round(abs_bbox);
abs_bbox(:, 3) = min(abs_bbox(:, 3), w);
abs_bbox(:, 4) = min(abs_bbox(:, 4), h);
abs_bbox(:, 1) = max(abs_bbox(:, 1), 1);
abs_bbox(:, 2) = max(abs_bbox(:, 2), 1);