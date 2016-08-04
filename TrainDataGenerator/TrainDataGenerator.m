clear all
close all
clc

%% Path parameters
box_size    = 25;
curr_folder = pwd;

targ_folder = [curr_folder '\Data\'];
imag_folder = [curr_folder '\Data\Blobs\'];

imag_folder_struct = dir(imag_folder);
totImages          = length(imag_folder_struct(not([imag_folder_struct.isdir])));

%% Data
%Reading
Data = zeros(totImages,box_size,box_size,3,'uint8');

for image_nbr = 1:totImages
    image_name = [imag_folder 'blob' num2str(image_nbr) '.jpeg'];
    image_file = imread(image_name);
    Data(image_nbr, :, :,:) = image_file;        
end

%% Targets
Target= (csvimport([targ_folder 'Target.csv']));

%Delete header
Target{1}=[];

%Convert into normal array
Target = cell2mat(Target);

%% Saving 
save([targ_folder   'Data.mat'], 'Data')
save([targ_folder 'Target.mat'], 'Target')

disp('Finished')
