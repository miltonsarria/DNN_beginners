function [ cost, grad ] = stackedDNNCost(theta, netconfig, datai, datao)
                                         
% stackedAECost: Takes a trained NN softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoders and the nn
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% datai: Our matrix containing the training data as columns.  So, datai(:,i) is the i-th training example. 
% datao: Our matrix containing the target data as columns.  So, datao(:,i) is the i-th target example. 
%% Unroll Theta parameters
stack = params2stack(theta, netconfig);
% You will need to compute the following gradients
%for d = 1:numel(stack)
%    stackgrad{d}.w = zeros(size(stack{d}.w));
%    stackgrad{d}.b = zeros(size(stack{d}.b));
%end
%cost = 0; % You need to compute this
% You might find these variables useful
M = size(datai, 2);
%% 
W1=stack{1}.w; b1=stack{1}.b;
W2=stack{2}.w; b2=stack{2}.b;
W3=stack{3}.w; b3=stack{3}.b;
W4=stack{4}.w; b4=stack{4}.b;

%% forward
a2 =sigmoid(bsxfun(@plus, W1*datai, b1));   %features 1
a3 =sigmoid(bsxfun(@plus, W2*a2, b2));      %features 2
a4 =sigmoid(bsxfun(@plus, W3*a3, b3));      %features 3
%output of the nn
a5 =bsxfun(@plus, W4*a4, b4);               %linear output
%cost function
cost=1/M*sum(1/2*sum((a5-datao).^2));
%% backward
delta_5=-(datao-a5); %-(datao-a5).*(a5.*(1-a5));  %output as sigmoid or linear
delta_4=(W4'*delta_5).*(a4.*(1-a4));
delta_3=(W3'*delta_4).*(a3.*(1-a3));
delta_2=(W2'*delta_3).*(a2.*(1-a2));

%gradients
stackgrad{1}.w=1/M*delta_2*datai';
stackgrad{2}.w=1/M*delta_3*a2';
stackgrad{3}.w=1/M*delta_4*a3';
stackgrad{4}.w=1/M*delta_5*a4';

stackgrad{1}.b=1/M*sum(delta_2,2);
stackgrad{2}.b=1/M*sum(delta_3,2);
stackgrad{3}.b=1/M*sum(delta_4,2);
stackgrad{4}.b=1/M*sum(delta_5,2);
%% Roll gradient vector
grad = stack2params(stackgrad);

end
% sigmoid function
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
