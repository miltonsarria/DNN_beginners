%% In this exercise: train two autoencoders, then a softmax model to classify
%mnist images
clear all
clc
close all
addpath minFunc/
addpath mnistHelper/
inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       
%%======================================================================
%% STEP 1: Load data from the MNIST database
%  This loads our training data from the MNIST database files.
% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
visibleSize = inputSize;   % number of input units 
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

 [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                    visibleSize, hiddenSizeL1, ...
                                    lambda, sparsityParam, ...
                                    beta, trainData), ...
                               sae1Theta, options);
%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%features from fisrt autoencoder
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
visibleSize = hiddenSizeL1;   % number of input units 
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);
%%======================================================================
%% STEP 3: Train the softmax classifier
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                       hiddenSizeL1, sae1Features);
%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
options.maxIter     = 100;
softmaxModel  = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
%%======================================================================
%% STEP 5: Finetune softmax model
% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];
%% -------
% finetune
options.Method = 'lbfgs';
options.maxIter = 400;
visibleSize = hiddenSizeL1;   % number of input units 
hiddenSize=hiddenSizeL2;
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAESoftCost(p, visibleSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                              stackedAETheta, options);

%%======================================================================
%% STEP 6: Test 
% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10
[pred] = stackedAESoftPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);
acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAESoftPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
%matlabpool close;
