clear all
close all
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
box_size    = 31;
curr_folder = pwd;

targ_folder = [curr_folder '\Data\'];
imag_folder = [curr_folder '\Data\Blobs\'];

imag_folder_struct = dir(imag_folder);

totImages          = length(imag_folder_struct(not([imag_folder_struct.isdir])));
if tf(1)==1 || tf(2)==1 % Veirfy if it is a lowercase or uppercase
    totImages          = length(imag_folder_struct(not([imag_folder_struct.isdir])))-1;
end

%% Training and Validation data
perc       = 0.7;
trainIndex  = floor(perc*totImages);
validIndex  = totImages-trainIndex;

% Preallocating
trainData = zeros(trainIndex,1,box_size,box_size,'uint8');
validData = zeros(validIndex,1,box_size,box_size,'uint8');

% Train data
for image_nbr = 1:trainIndex
    image_name = [imag_folder 'blob' num2str(image_nbr) '.jpeg'];
    image_file = imread(image_name);
    trainData(image_nbr, :, :,:) = rgb2gray(image_file);        
end

% Validation data
for image_nbr = validIndex:totImages
    image_name = [imag_folder 'blob' num2str(image_nbr) '.jpeg'];
    image_file = imread(image_name);
    validData(image_nbr, :, :,:) = rgb2gray(image_file);        
end

%% Training and Validation Targets
Target= (csvimport([targ_folder 'Target.csv']));

%Delete header
Target{1}=[];

%Convert into normal array
Target = cell2mat(Target);

%Spliting targets
trainTarget = Target(1:trainIndex);
validTarget = Target(trainIndex+1:end);

%% Saving 
save([targ_folder   'trainData.mat'], 'trainData')
save([targ_folder   'validData.mat'], 'validData')

save([targ_folder 'trainTarget.mat'], 'trainTarget')
save([targ_folder 'validTarget.mat'], 'validTarget')

disp('Finished')
