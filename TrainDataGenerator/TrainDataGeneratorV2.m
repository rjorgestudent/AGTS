clc
clearvars -except allresults

minSampleRadius = 2.5;
maxSampleRadius = 200;
%% Variable check
if ~isempty(find(cellfun(@isempty,allresults.output), 1))
    [emptyIdx, ~] = find(cellfun(@isempty,allresults.output));
    allresults(emptyIdx,:) = [];
end
%% Read variables from batchall
allResults     = allresults;
totImages      = size(allResults,1);
output         = cell2mat(allResults.output);
boxSizes       = output(:,3);
imPaths        = allResults.fileName;

%% Get boxSize
boxSize = round(mean(boxSizes));
    % boxSize has to be odd number
    if ~bitget(boxSize,1) %even
         boxSize = boxSize+1;
    end 
    
%% Generate extra coords
marginBorder = floor(boxSize/2)+1;

for i=1: length(allResults.output)
% Get image
im = rgb2gray(imread(imPaths{i}));

% Generate distanceMap
distance2Positive =  makeDistanceMap(im, allResults.output{i}(:,1:2));

%%%% POSITIVE %%%%%
% Get positions of new samples
[row,col] = find(distance2Positive<minSampleRadius);

% Remove new samples that are close to the borders
col = col(row > marginBorder & row< (size(im,1) - marginBorder));
row = row(row > marginBorder & row< (size(im,1) - marginBorder));
row = row(col > marginBorder & col< (size(im,2) - marginBorder));
col = col(col > marginBorder & col< (size(im,2) - marginBorder));

nbrPositives = length(row);

% Add new samples
allResults.output{i}=[col, row];
allResults.target{i}=true(size(col));

%%%% NEGATIVE %%%%%%
offset  = length(allResults.target{i});
    while length(allResults.target{i}) - offset < 2*nbrPositives
        x = randi([marginBorder, size(im,2)-marginBorder],1,1);
        y = randi([marginBorder, size(im,1)-marginBorder],1,1);

        d = distance2Positive(y,x);

        if d > minSampleRadius || d< maxSampleRadius
            allResults.output{i}=[allResults.output{i}; x,y];
            allResults.target{i}=[allResults.target{i}; false];
        end
    end
end

%% Initialization
% Preallocating
    allBlobs = zeros(0,1,boxSize,boxSize,'uint8');
    traBlobs = zeros(0,1,boxSize,boxSize,'uint8');
    valBlobs = zeros(0,1,boxSize,boxSize,'uint8');
    allTarge = false(0,1);
    traTarge = false(0,1);
    valTarge = false(0,1);
    
%% Samples generation
    
    % Samples NormalView
    for i=1:totImages
        allBlobs = blobsNormal(imPaths{i},allResults.output{i},boxSize,allBlobs);
        allTarge = [allTarge; allResults.target{i}];
    end
    
    % Samples HorizontalMirror
     for i=1:totImages
        allBlobs = blobsHorizMirr(imPaths{i},allResults.output{i},boxSize,allBlobs);
        allTarge = [allTarge; allResults.target{i}];
     end        
    
     % Samples rotation +5°
     for i=1:totImages
        allBlobs = blobsRotated(imPaths{i},allResults.output{i},boxSize,allBlobs, 5);
        allTarge = [allTarge; allResults.target{i}];
     end 
     % Samples rotation -5°
     for i=1:totImages
        allBlobs = blobsRotated(imPaths{i},allResults.output{i},boxSize,allBlobs, -5);
        allTarge = [allTarge; allResults.target{i}];
     end
%% Shuffle data
suffleIdx = randperm(size(allTarge,1))';
allBlobs  = allBlobs(suffleIdx, :, :, :);
allTarge  = allTarge(suffleIdx);
%% Save
[FileName,PathName] = uiputfile('agtsNoche');

save([PathName FileName 'DATA' datestr(now,'ddmmmyy') '.mat'   ], 'allBlobs')
save([PathName FileName 'TARGET' datestr(now,'ddmmmyy') '.mat'   ], 'allTarge')
