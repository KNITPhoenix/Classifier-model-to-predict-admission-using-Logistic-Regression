function p = predict(theta, X)
m = size(X, 1); % Number of training examples
p = zeros(m, 1);

for i=1:m,
	if(sigmoid(theta' * X(i,:)')>=0.5)
		p(i)=1;
	else
		p(i)=0;
	endif
end;

end
