clear all
close all
clc

curr_folder = dir;
imag_folder = [curr_folder '\Data\Blobs'];
totImages   = length(curr_folder(not([curr_folder.isdir])))-2;
%%
for image_nbr = 1:totImages
    image_name = ['avocados_' num2str(image_nbr) '.jpeg'];
    image_file = imread(image_name);
    data(image_nbr, :, :, :) = image_file;        
end