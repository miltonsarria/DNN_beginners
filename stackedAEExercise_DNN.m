%In this exercise: train two autoencoders, then a softmax model to classify
%mnist images
clear all
clc
close all
addpath(genpath('/home/sarria/Documents/MATLAB/autoencoder'))

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       
% nworkers = 3;
% nworkers = min(nworkers, feature('NumCores'));
% isopen = matlabpool('size')>0;
% if ~isopen, matlabpool(nworkers); end

%%======================================================================
%% STEP 1: Load data from the MNIST database
%  This loads our training data from the MNIST database files.
% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
%%======================================================================
%% STEP 2: Train the NN with the DNN toolbox
%  Randomly initialize the parameters
rand('state',0)
sae = saesetup([inputSize hiddenSizeL1 hiddenSizeL2]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0;
sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0;

opts.numepochs =   100;
opts.batchsize = 100;

sae = saetrain(sae, trainData', opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([inputSize hiddenSizeL1 hiddenSizeL2 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};

% Train the FFNN
opts.numepochs =   10;
opts.batchsize = 100;

numClasses=10;
mask=repmat((1:numClasses)',1,length(trainLabels))==repmat(trainLabels(:)',numClasses,1);
Labels=double(mask);


nn = nntrain(nn, trainData', Labels', opts);


testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10
mask=repmat((1:numClasses)',1,length(testLabels))==repmat(testLabels(:)',numClasses,1);
Labels=double(mask);

[er, bad] = nntest(nn, testData', Labels');
%%% here another way of thesting the nn
nn.testing = 1;
nn = nnff(nn, testData', zeros(size(testData',1), nn.size(end)));
nn.testing = 0;
out=nn.a{end};
[p,v]=max(out');
acc=sum(testLabels==v')/length(v)*100;

fprintf('Test accuracy: %0.3f%%\n', 100-er * 100);

%the results with my implementation
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
%matlabpool close;
