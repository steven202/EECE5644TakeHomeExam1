% Expected risk minimization with 2 classes
clear; close all; %clc;

rng('default');
rng(1);

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma(:,:,1) = [1, -0.9;-0.9,1]; Sigma(:,:,2) = [1, 0.9;0.9, 1];
p = [0.8,0.2]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
ratio = evalGaussian(x,mu(:,2),Sigma(:,:,2))-evalGaussian(x,mu(:,1),Sigma(:,:,1));
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
% socre < gamma - 0 
% score >=gamma - 1
decision = (discriminantScore >= log(gamma)); 

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated
%%%%%%%%%%%%%%%%%%%%%%%
% plot the minimum probablity of error curve. 

figure(6), clf; 
dim0=1000;
dim1=10;
gamma0=linspace(0,dim1,dim0);

p_error = zeros(1,dim0);
for i = 1:dim0
   decision = (discriminantScore>=log(gamma0(i)));
   
   ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
   ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
   
   p_error(i) = [p10,p01]*Nc'/N;
end
   
plot(gamma0, p_error, '-','LineWidth',2);

col = find(p_error==min(p_error),1,'first');
gamma0(col);
%%%%%%%%%%%%%
% compute the error point
decision = (discriminantScore>=log(gamma));

ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative

p_error_point = [p10,p01]*Nc'/N;
% mark the point
hold on;
plot(gamma0(col),p_error_point,'r*'); 
text(gamma0(col),p_error_point,'(3.7738, 0.0817)','FontSize',13);
title('minimum probablity of error curve'),
xlabel('threshold gamma'), ylabel('probability of error');
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% finish %%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%%%%%%
figure(4), clf;
gammas = linspace(0, dim1,dim0);
gammas = [gammas, inf];
threshold=zeros(2,dim0+1);

for i=1:dim0+1
    decision_temp = (discriminantScore >= log(gammas(i))); 
    ind11_temp = find(decision_temp==1 & label==1); p11_temp = length(ind11_temp)/Nc(2); % probability of true positive
    ind10_temp = find(decision_temp==1 & label==0); p10_temp = length(ind10_temp)/Nc(1); % probability of false positive
    threshold(1,i)=p11_temp;
    threshold(2,i)=p10_temp;
end

Y = threshold(1,:);
X = threshold(2,:);
plot(X,Y,'-','LineWidth',2), hold on;
title('Roc curve'),
xlabel('probability false positive'), ylabel('probability of true positive');

decision_temp = (discriminantScore >= log(gamma0(col)));
ind11_temp = find(decision_temp==1 & label==1); p11_temp = length(ind11_temp)/Nc(2); % probability of true positive
ind10_temp = find(decision_temp==1 & label==0); p10_temp = length(ind10_temp)/Nc(1); % probability of false positive
%DO NOT USE THIS LINE!@!!!!!!!!!!!!!!!!
%scatter(p10_temp, p11_temp);
%plot(p10_temp, p11_temp, '--rs','LineWidth', 2, 'MarketEdgeColor', 'k', 'MarketFaceColor', 'g', 'MarkerSize', 15), axis equal;
hold on;
plot(p10_temp, p11_temp,'*r');
text(p10_temp,p11_temp,'(0.0180, 0.6570)','FontSize',13);
fprintf('threshold gamma(from calculation): %d\n',gamma);
fprintf('threshold gamma(from graph): %d\n',gamma0(col));
fprintf('minimum probability of error: %d\n', p_error_point);


