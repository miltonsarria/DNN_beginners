function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 


epsilon=10e-6;
N_theta=length(numgrad);

for ii=1:N_theta;
    e=zeros(size(theta)); e(ii)=1;  
    J1=J(theta+epsilon*e);
        
    J2=J(theta-epsilon*e);
    
    numgrad(ii)=(J1-J2)/(2*epsilon); 
    por=ii/N_theta*100; 
    clc
    fprintf(1,'computing numerical gradient.... %f \n',por)
end






%% ---------------------------------------------------------------
end
