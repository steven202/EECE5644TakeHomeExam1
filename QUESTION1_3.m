% Expected risk minimization with 2 classes
clear; close all; %clc;

rng('default');
rng(1);

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma(:,:,1) = [1, -0.9;-0.9,1]; Sigma(:,:,2) = [1, 0.9;0.9, 1];
p = [0.7,0.3]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

lambda = [0 1;1 0]; % loss values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Appending LDA to the ERM code for TakeHomeQ3...
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);


%%%%%%%
% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector

% Linearly project the data from both categories on to w
yLDA = wLDA'*x;
%%%%%%%

% % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% [V,D] = eig(inv(Sw)*Sb); 
% % equivalently alpha w  = inv(Sw) Sb w
% [~,ind] = sort(diag(D),'descend');
% wLDA = V(:,ind(1)); % Fisher LDA projection vector
% yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
% wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
% yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

%%%%%%%%%%%%%% plot LDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
tau = 0;
decisionLDA = (yLDA >= 0);

decision = (yLDA >= tau);
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
tau0=linspace(-dim1,dim1,dim0);
tau0 = [-inf,tau0,inf];
p_error = zeros(1,dim0+2);
for i = 1:dim0+2
   decision = (yLDA>=tau0(i));
   
   ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
   ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
   
   p_error(i) = [p10,p01]*Nc'/N;
end
   
plot(tau0, p_error, '-','LineWidth',2);

col = find(p_error==min(p_error), 1,'first');
fprintf("tau is: %s\n", tau0(col));
%%%%%%%%%%%%%
% compute the error point
decision = (yLDA>=tau0(col));

ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative

p_error_point = [p10,p01]*Nc'/N;
% mark the point
hold on;
plot(tau0(col),p_error_point,'r*'); 
text(tau0(col),p_error_point,'(3.0130, 0.2947)','FontSize',13);
title('minimum probablity of error curve'),
xlabel('threshold tau'), ylabel('probability of error');
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% finish %%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% ROC Curve %%%%%%%%%%%%%%%%%%%%%
figure(4), clf;

taus = linspace(-dim1,dim1,dim0);
taus = [-inf,taus,inf];
threshold=zeros(2,dim0+2);

for i=1:dim0+2
    decision_temp = (yLDA>=taus(i));
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

decision_temp = (yLDA>=taus(col));
ind11_temp = find(decision_temp==1 & label==1); p11_temp = length(ind11_temp)/Nc(2); % probability of true positive
ind10_temp = find(decision_temp==1 & label==0); p10_temp = length(ind10_temp)/Nc(1); % probability of false positive
%DO NOT USE THIS LINE!@!!!!!!!!!!!!!!!!
%scatter(p10_temp, p11_temp);
%plot(p10_temp, p11_temp, '--rs','LineWidth', 2, 'MarketEdgeColor', 'k', 'MarketFaceColor', 'g', 'MarkerSize', 15), axis equal;
hold on;
plot(p10_temp, p11_temp,'*r');
text(p10_temp,p11_temp,'(5.6746e-04, 0.0027)','FontSize',13);

fprintf('threshold tau: %d\n',tau0(col));
fprintf('minimum probability of error: %d\n', p_error_point);