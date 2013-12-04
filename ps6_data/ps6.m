clear; close all

% Example code starts from here
% Load the data set
load('faces.csv');

% display the first face
colormap gray
imagesc(reshape(faces(1, :), 64, 64));

% Your code starts from here ...

% a. Randomly display a face
% STUDENT CODE TODO

% b. Compute and display the mean face
% STUDENT CODE TODO

% c. Centralize the faces by substracting the mean
% STUDENT CODE TODO

% d. Perform SVD (you may find the function svd useful)
% STUDENT CODE TODO

% e. Show the first 10 priciple components
% STUDENT CODE TODO

% f. Visualize the data by using first 2 principal components using the function "visualize.m"
% STUDENT CODE TODO

% g. Plot the proportion of variance explained
% STUDENT CODE TODO

% h. Face reconstruction using 5, 10, 25, 50, 100, 200, 300, 399 principal components
% STUDENT CODE TODO

% i. Plot the reconstruction error for k = 5, 10, 25, 50, 100, 200, 300, 399 principal components
%    and the sum of the squares of the last n-k (singular values)
%    [extra credit]
% STUDENT CODE TODO
