function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


M=theta*data;
M = bsxfun(@minus, M, max(M, [], 1));

e_M=exp(M);
sum_E=sum(e_M);
e_M_n=(e_M./repmat(sum_E,size(e_M,1),1));
L_M=log(e_M_n);

mask=repmat([1:numClasses]',1,length(labels))==repmat(labels',numClasses,1);
cost=-1/length(labels)*sum(sum(mask.*L_M))+lambda/2*sum(sum(theta.^2));

for ii=1:numClasses
    x=-1/length(labels)*sum(data.*(repmat(mask(ii,:)-e_M_n(ii,:),inputSize,1)),2);    
    thetagrad(ii,:)=(x+lambda*theta(ii,:)')';    
end

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

