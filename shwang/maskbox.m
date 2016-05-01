function [newboxes,scores]=maskbox(heatmap,boxes,a)
sizeofbb = (boxes(:,3)-boxes(:,1)).*(boxes(:,4)-boxes(:,2));
for i=1:size(boxes)
    t = heatmap(boxes(i,2):boxes(i,4),boxes(i,1):boxes(i,3));
    ov = sum(sum(t));
    if a
        scores(i)= ov/sizeofbb(i);
    else
        scores(i)= ov;
    end

end
[scores,ids]=sort(scores,'descend');
newboxes = boxes(ids,:);
