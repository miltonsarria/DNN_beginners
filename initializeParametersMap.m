function theta = initializeParametersMap(inputSize,hiddenSize, outputSize)

%% Initialize parameters randomly based on layer sizes.
r  =  0.12;   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, inputSize) * 2 * r - r;
W2 = rand(outputSize, hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize, 1);
b2 = zeros(outputSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

