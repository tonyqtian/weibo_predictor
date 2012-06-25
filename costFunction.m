function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%g = 1 ./ (1 + exp(- X * theta));
%sigf = sigmoid(X * theta);
expth = exp(- X * theta);
siglog = -log(1 + expth);
signlog = siglog + log(expth);
%for i = 1:m
%	if sigf(i) == 1
%		signlog(i) = 0;
%	end
%	if sigf(i) == 0
%		siglog(i) = 0;
%	end
%end
%for i = 1:m
%	if y == 1
%		signlog(i) = 0;
%	end
%	if y == 0
%		siglog(i) = 0;
%	end
%end
%siglog
%signlog
J = - ( y' * siglog + (1-y)' * signlog ) / m;
%costp = (sigmoid(X * theta) - y)'; %'
%for i = 1:size(theta)
%	grad(i) = costp * X(:,i);
%end
grad = X' * (sigmoid(X * theta) - y) ./m; %'
%B = ones(length(y),1);
%K = (B - y - y.* expth) ./ (B + expth);
%grad = X' * K ./ m; %'

% =============================================================

end
