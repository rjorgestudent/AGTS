clear all
%close all
clc

%% Thumbs file control
prompt = 'Thumbs.db existe? [y/n]';
answer = input(prompt,'s');

if isempty(answer)
    answer = 'Y';
end
possibleAnswers = {'y','Y'};
tf              = strcmp(answer,possibleAnswers);

%% Path parameters

curr_folder = pwd;

imag_folder        = [curr_folder '\Comprimidas\'];
imag_folder_struct = dir(imag_folder);

totImages          = length(imag_folder_struct(not([imag_folder_struct.isdir])));  
if tf(1)==1 || tf(2)==1
    totImages          = length(imag_folder_struct(not([imag_folder_struct.isdir])))-1; % File Thumbs.db of Windows
end
%% Data

for image_nbr = 1:totImages
    % Displaying progress
    Percentage = floor((image_nbr/totImages)*100);
    disp(['Progress: ' num2str(Percentage) '%']);
    
    % Opening file
    image_name = [imag_folder 'tree (' num2str(image_nbr) ').jpg'];
    imOriginal = imread(image_name);
    
    % Removing sky
    imGrayScal = double(rgb2gray(imOriginal));
    T1         = opthr(imGrayScal);
    Mask1      = uint8(bwmorph(imGrayScal > T1, 'open'));    
    imNoSky    = imOriginal.*repmat(Mask1,[1,1,3]);
    
    % Removing ground
    imLabMap   = double(rgb2lab(imNoSky));
    
    imSpaceA   = imLabMap(:,:,2);
    T2         = opthr(imSpaceA);
    Mask2      = uint8(bwmorph(imSpaceA < T2, 'open')); 
    
    imSpaceB   = imLabMap(:,:,3);
    T3         = opthr(imSpaceB);
    Mask3      = uint8(bwmorph(imSpaceB > T3, 'open')); 
    
    Mask       = Mask2.*Mask3;
    imNoGnd    = imNoSky.*repmat(Mask,[1,1,3]);
    
    
    
    % Saving 
    imwrite(imNoGnd, [curr_folder   '\Procesadas\tree' num2str(image_nbr) '.jpg'])
    
clc         
end

%%
disp('Finished')
