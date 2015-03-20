function [ cost, grad ] = stackedAESoftCost(theta, visibleSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));
%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
W1=stack{1}.w; b1=stack{1}.b;
W2=stack{2}.w; b2=stack{2}.b;
%% forward
a2=sigmoid(W1*data+repmat(b1,1,M)); %features 1
a3=sigmoid(W2*a2+repmat(b2,1,M));   %features 2

a4 =softmaxTheta*a3;                %output of the softmax
a4 = bsxfun(@minus, a4, max(a4, [], 1));

e_M=exp(a4);
sum_E=sum(e_M);
e_M_n=(e_M./repmat(sum_E,size(e_M,1),1));
L_M=log(e_M_n);

mask=repmat((1:numClasses)',1,length(labels))==repmat(labels(:)',numClasses,1);
cost=-1/length(labels)*sum(sum(mask.*L_M))+lambda/2*sum(sum(softmaxTheta.^2));

D_J=softmaxTheta'*(mask-e_M_n);
delta_3=-D_J.*(a3.*(1-a3));
delta_2=(W2'*delta_3).*(a2.*(1-a2));


stackgrad{1}.w=1/M*delta_2*data';
stackgrad{2}.w=1/M*delta_3*a2';

%for ii=1:M
%    W1grad=W1grad+delta_2(:,ii)*a1(:,ii)';
%    W2grad=W2grad+delta_3(:,ii)*a2(:,ii)';
%end
stackgrad{1}.b = 1/M*sum(delta_2,2);
stackgrad{2}.b = 1/M*sum(delta_3,2);

for ii=1:numClasses
    x=-1/length(labels)*sum(a3.*(repmat(mask(ii,:)-e_M_n(ii,:),size(a3,1),1)),2);    
    softmaxThetaGrad(ii,:)=(x+lambda*softmaxTheta(ii,:)')';    
end



% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
