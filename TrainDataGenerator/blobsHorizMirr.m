function allBlobs = blobsHorizMirr(imPath,coords,boxSize,allBlobs)
%BLOBSNORMAL stores the patches of the image in normal persective
%   Detailed explanation goes here
im = rgb2gray(imread(imPath));
im = flip(im,2);

initSize = size(allBlobs,1);
for patchIdx = 1: size(coords,1)
    patchCurrent = imcrop(im,[coords(patchIdx,1)-((boxSize-1)/2), coords(patchIdx,2)-((boxSize-1)/2), boxSize-1,boxSize-1 ]); 
    allBlobs(initSize + patchIdx, :, :,:) = patchCurrent;
    
end


end

