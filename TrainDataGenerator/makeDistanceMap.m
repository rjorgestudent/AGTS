function distanceMap = makeDistanceMap( im, coords )
%DISTANCEMAP computes the distance between the centers of the fruits
%   Detailed explanation goes here

f = false(size(im));
[v,u] = size(im);
zeroVector = zeros(size(coords,1),1); 

x = min(max(zeroVector, coords(:,1)), u-1);
y = min(max(zeroVector, coords(:,2)), v-1);

f(sub2ind([v,u], y, x))= true;

%for idx=1:size(coords,1)
%   x = int(min(max(0, coords(idx,1)), size(im,2)-1));
%    y = int(min(max(0, coords(idx,2)), size(im,1)-1));
%    
%    f(x,y) = false;    
% end


distanceMap = bwdist(f);


end

