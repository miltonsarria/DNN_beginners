%% training and Visualization: sparseAutoencoder
%% STEP 0: initial paramaters 
clear all
clc
close all
addpath minFunc/
addpath mnistHelper/
visibleSize = 8*8;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.01;% desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       
%%======================================================================
%% STEP 1: Implement sampleIMAGES
%  the display_network command displays a random sample of 200 patches from the dataset

patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),200,1)),8);
%%======================================================================
%% STEP 2: train the sparse autoencoder with minFunc (L-BFGS).
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);
%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
%%======================================================================
%% STEP 3: Visualization 
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 
print -djpeg weights.jpg   % save the visualization to a file 
