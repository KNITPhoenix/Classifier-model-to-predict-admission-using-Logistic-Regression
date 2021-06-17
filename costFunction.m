function [J, grad] = costFunction(theta, X, y)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

a=0;
for i=1:m,
	a=a+(-(y(i)*log(sigmoid(theta' * X(i,:)')))-((1-y(i))*log(1-sigmoid(theta' * X(i,:)'))));
end;
J=(1/m)*a;

grad=(1/m)*(X'*((sigmoid(theta' * X'))' - y));

end
