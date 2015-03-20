function [output] = DNN_AEPredict(Theta,netconfig,datai)

%% Unroll Theta parameters
stack = params2stack(Theta, netconfig);
% You will need to compute the following gradients
%for d = 1:numel(stack)
%    stackgrad{d}.w = zeros(size(stack{d}.w));
%    stackgrad{d}.b = zeros(size(stack{d}.b));
%end
%cost = 0; % You need to compute this
% You might find these variables useful
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
output =bsxfun(@plus, W4*a4, b4);           %linear output
% -----------------------------------------------------------
end
% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
