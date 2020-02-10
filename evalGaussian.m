function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);

%function y = evalGaussian(x,mu,Sigma)
y = mvnpdf(x',mu',Sigma)';
disp(sum((g-y).^2));
%g=y




% function g = evalGaussian(x,mu,Sigma)
% [n,N] = size(x);
% g=zeros(N);
% C = ((2*pi)^n * det(Sigma))^(-1/2);
% for i = 1:N 
%     E = -0.5*(x(:,i)-mu)'*(inv(Sigma)*(x(:,i)-mu));
%     g(i)=C*exp(E);
% end
