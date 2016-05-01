function box_visualize(im,boxes,Num,clr,isedge)
% figure
imshow(im)
[w,h,~]=size(im);
Num = min(size(boxes,1),Num);
for i=1:Num
    U=boxes(i,:);
    x =[U(1),U(1),U(3),U(3)];
    y =[U(2),U(4),U(4),U(2)];
    
    % 0.01 transparency ratio
    if isedge==1
        patch(x,y,'r','facealpha',0.01,'LineWidth',0.1,'facecolor',clr);xlim([-100,h+100]);ylim([-100,w+100]);
    elseif isedge==0
        patch(x,y,'r','facealpha',0.01,'EdgeColor','white','facecolor',clr);xlim([-100,h+100]);ylim([-100,w+100]);
    else
        patch(x,y,'r','facealpha',0.01,'EdgeColor','none','facecolor',clr);xlim([-100,h+100]);ylim([-100,w+100]);
    end
end
% ,
