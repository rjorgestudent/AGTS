clear all
close all
clc

curr_folder = dir;
imag_folder = [pwd '\Comprimidas'];
totImages   = length(curr_folder(not([curr_folder.isdir])))-2;
%%
for image_nbr = 1:totImages
    image_name = ['DSC' num2str(image_nbr) '.jpeg'];
    image_file = imread(image_name);
    data(image_nbr, :, :, :) = image_file;        
end