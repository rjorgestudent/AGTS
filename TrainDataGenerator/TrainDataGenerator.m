clear all
close all
clc

curr_folder = pwd;

targ_folder = [curr_folder '\Data\'];

imag_folder = [curr_folder '\Data\Blobs\'];
imag_struct = dir(imag_folder);
totImages   = length(imag_struct(not([imag_struct.isdir])));

%% Data
%Reading
data = zeros(totImages,24,24,3,'uint8');

for image_nbr = 1:totImages
    image_name = [imag_folder 'blob' num2str(image_nbr) '.jpeg'];
    image_file = imread(image_name);
    data(image_nbr, :, :,:) = image_file;        
end

%% Targets
Target= (csvimport([targ_folder 'Target.csv']));

%Delete header
Target{1}=[];

%Convert into normal array
Target = cell2mat(Target);

%% Saving 
save([targ_folder 'Data.mat'], 'data')
save([targ_folder 'Target.mat'], 'Target')
