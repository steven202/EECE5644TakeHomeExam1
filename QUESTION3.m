%%%%%% The code is partially borrowed from the shared folder provided by the Professor Deniz.
%%%%%% Some of the code is from volunteers in the shared folder. 
%%%%%% Many thanks to professor and volunteers. 
% Expected risk minimization with 2 classes
clear; close all; %clc;

rng('default');
rng(1);

n = 1; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = -2; mu(:,2) = 2;
Sigma(:,:,1) = 1; Sigma(:,:,2) = 1;
p = [0.5,0.5]; % class priors for labels 0 and 1 respectively
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
text(gamma0(col),p_error_point,'(1.0811, 0.0222)','FontSize',13);
title('minimum probablity of error curve'),
xlabel('threshold gamma'), ylabel('probability of error');
%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% finish %%%%%%%%%%%%%%%%%

fprintf('threshold gamma(from calculation): %d\n',gamma);
fprintf('threshold gamma(from graph): %d\n',gamma0(col));
fprintf('minimum probability of error: %d\n', p_error_point);

%%%%%%%%%%%
%%% calculate integral
% copy from https://stats.stackexchange.com/questions/103800/calculate-probability-area-under-the-overlapping-area-of-two-normal-distributi
mu1=-2; 
mu2=2;
c1=1;
c2=1;
acc=(mu1-mu2)^2+2*(c1^2-c2^2)*log(c1/c2);
acd=mu1*c2+c1*sqrt(acc);
acr=mu2*(c1^2)-c2*(acd);
c=acr/(c1^2-c2^2+0.00000000001);%prevent dividing by 0; 
overlap_area=1- 1/2*erf((c-mu1)/(sqrt(2)*c1))+1/2*erf((c-mu2)/(sqrt(2)*c2));
fprintf("overlap_area is: %d\n", overlap_area);
error_from_integral = overlap_area/2;
fprintf("error_from_integral is: %d\n",error_from_integral);


figure(7), clf; 
x =-10:.1:10;
y = normpdf(x,-2,1);
z = normpdf(x,2,1);
plot(x,y)
hold on
plot(x,z)
title('probability density function'),
xlabel('x'), ylabel('probability density');

legend('mean = -2','mean = 2')
% Mean = -2;
% Standard_Deviation = 1;
% lims = [-1000 1000];
% cp = normcdf(lims, Mean, Standard_Deviation);
% Prob = cp(1) - cp(2);
% 
% Mean2 = 2;
% Standard_Deviation2 = 1;
% lims2 = [-1000 1000];
% cp2 = normcdf(lims2, Mean2, Standard_Deviation2);
% Prob2 = cp2(1) - cp2(2);
% result=0;
% delta=0.001;
% for i = -1000:delta:1000
%    if normpdf(i, mu1,c1)<delta
%       result= result+ normpdf(i, mu2,c2);
%    elseif normpdf(i, mu2,c2)<delta
%       result= result+ normpdf(i, mu1,c1);
%    else
%       result= result+ max(normpdf(i, mu1,c1),normpdf(i, mu2,c2));
%    end
% end
% result=result*delta;
% disp(result);