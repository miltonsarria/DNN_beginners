function [cost,grad] = sparseMaping(theta, inputSize,hiddenSize, outputSize,lambda, sparsityParam, beta, datai,datao)
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
W2 = reshape(theta(hiddenSize*inputSize+1:hiddenSize*inputSize+hiddenSize*outputSize), outputSize, hiddenSize);

b1 = theta(hiddenSize*inputSize+hiddenSize*outputSize+1:hiddenSize*inputSize+hiddenSize*outputSize+hiddenSize);
b2 = theta(hiddenSize*inputSize+hiddenSize*outputSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values) Here, we initialize them to zeros. 
%cost = 0; W1grad = zeros(size(W1)); W2grad = zeros(size(W2)); b1grad = zeros(size(b1)); b2grad = zeros(size(b2));
%% forward
rho=sparsityParam;
M=size(datai,2);
%hidden  
a2=sigmoid(bsxfun(@plus, W1*datai, b1));
%a2=tanh(bsxfun(@plus, W1*datai, b1));
%output, can be sigmoid, tanh or linear
%hwb=sigmoid(bsxfun(@plus, W2*a2, b2)); 
%hwb=tanh(bsxfun(@plus, W2*a2, b2));
hwb=bsxfun(@plus, W2*a2, b2);

term1=1/M*sum(1/2*sum((hwb-datao).^2));
term2=lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));

rho_hat=mean(a2,2);
term3=beta*sum(rho*log(rho./rho_hat)+(1-rho)*log((1-rho)./(1-rho_hat)));
cost=term1+term2+term3;
term_deltas=beta*(-rho./rho_hat+(1-rho)./(1-rho_hat));
%compute gradients
%delta_3=-(datao-hwb).*(hwb.*(1-hwb)); %output as sigmoid
%delta_3=-(datao-hwb).*(1-hwb.^2);     %output as tanh
delta_3=-(datao-hwb);                  %output as linear
delta_2=(W2'*delta_3+repmat(term_deltas,1,M)).*(a2.*(1-a2));%hiden as sigmoid
%delta_2=(W2'*delta_3+repmat(term_deltas,1,M)).*(1-a2.^2);%hiden as tanh
W1grad=delta_2*datai';    
W2grad=delta_3*a2';
W1grad=1/M*W1grad+lambda*W1;
W2grad=1/M*W2grad+lambda*W2;
b1grad=1/M*sum(delta_2,2);
b2grad=1/M*sum(delta_3,2);
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc)
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end
function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
