function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)
% visibleSize: the number of input units 
% hiddenSize: the number of hidden units 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our nxp matrix containing the training data.  So, data(:,i) is the i-th training example. 
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format.
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
%% forward
rho=sparsityParam;
M=size(data,2);
%hidden
a2=sigmoid(bsxfun(@plus, W1*data, b1));
%a2=tanh(W1*datai+repmat(b1,1,M));
%output 
hwb=sigmoid(bsxfun(@plus, W2*a2, b2));
%hwb=tanh(W2*a2+repmat(b2,1,M));
%hwb=W2*a2+repmat(b2,1,M);
%compute cost function
term1=1/M*sum(1/2*sum((hwb-data).^2));
term2=lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
rho_hat=mean(a2,2);
term3=beta*sum(rho*log(rho./rho_hat)+(1-rho)*log((1-rho)./(1-rho_hat)));
cost=term1+term2+term3;
term_deltas=beta*(-rho./rho_hat+(1-rho)./(1-rho_hat));
%compute gradients
delta_3=-(data-hwb).*(hwb.*(1-hwb)); %output as sigmoid
%delta_3=-(data-hwb).*(1-hwb.^2);     %output as tanh
%delta_3=-(data-hwb);                 %output as linear
delta_2=(bsxfun(@plus, W2'*delta_3, term_deltas)).*(a2.*(1-a2));%hiden as sigmoid
%delta_2=(W2'*delta_3+repmat(term_deltas,1,M)).*(1-a2.^2);%hiden as tanh

W1grad=delta_2*data';    
W2grad=delta_3*a2'; 

W1grad=1/M*W1grad+lambda*W1;
W2grad=1/M*W2grad+lambda*W2;
b1grad=1/M*sum(delta_2,2);
b2grad=1/M*sum(delta_3,2);
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc). 
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end
%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function
function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end