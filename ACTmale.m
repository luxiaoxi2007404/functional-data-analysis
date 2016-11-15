clc;clear

addpath('E:\fdaM')

%  ----------------------------------------------------------------
%  --------------------  载入男性的测试结果数据  --------------------
%  ----------------------------------------------------------------

nit =   60;
nex = 2115;
%直接读入数据有问题，采用按文本读入，再整理格式
fid = fopen('actm.txt','rt');
ACTmtest = reshape(fscanf(fid, '%s'), [nit,nex])';

%设置theta并等分
nq = 21;
thetamax = 2.8; thetamin = -2.8;
thetaq = linspace(thetamin, thetamax, nq)';
thetarng = [thetamin, thetamax];

%  ---------------------------  创建样条 -----------------------------

nbasis = 11;
norder = 4;
wbasis = create_bspline_basis(thetarng, nbasis);

% 样条在指定点处的值
phimat = getbasismatrix(thetaq, wbasis);
% 加惩罚项的在指定点出的值
Kmat =  eval_penalty(wbasis, 2);

%  ----------------------------  EM算法  ---------------------------
%%%% 初始化EM算法的P值
%      Function FirstStep computes the proportions of examinees 
%       associated with each value of thetaq that pass items
P0 = FirstStep(ACTmtest, thetaq);  %利用所有测试者的信息，得到不同theta下的P
W0 = log(P0./(1-P0));                
Wfd0 = data2fd(W0, thetaq, wbasis);  
coef0 = getcoef(Wfd0);   %基函数的系数

W = eval_fd(Wfd0, thetaq); % 估计方程在不同theta节点处的值
P = 1./(1+exp(-W0)); 
Q = 1 - P;  
Wfd = Wfd0; 
coef = coef0; 

lambda = 1e-1; % smoothing parameter lambda
penmat = lambda.*Kmat; % penalty matrix times lambda
iter = 0; % initialize iteration number
convtest = 1e-2; % convergence criterion
itermax = 60; % maximum number of iterations
F = 1e10; % initialize function value
Fold = F + 2*convtest; % initialize old function value

%%%% 计算权重
norder = 4; % Order of the B-spline
nbasis = nq + norder - 2; % Number of basis functions
wgtq = gausswgtBS(thetaq, nbasis, norder);

%%%% EM算法的估计过程
while Fold - F > convtest & iter < itermax
   if iter == 0, disp('It.    -log L   penalty    F     Fold - F'); end
   iter = iter + 1;
   Fold = F;

  %  --------  E step  ------------

  [N, CN, CP, L, CL] = Estep(ACTmtest, P, wgtq);
   
  %  compute penalized negative sum of marginal likelihoods
  logL = sum(log(L));
  pen  = sum(diag(coef' * penmat * coef));
  F = -logL + pen;
  %  print out F, which is being minimized
  fprintf('%g  ', [iter, -logL, pen, F, Fold - F]);  fprintf('\n');
   
  %   -------  Mstep  -------------
  
  coef = Mstep(CN, N, P, coef, phimat, penmat);
  % update Q by n matrix of probabilities 
  P = 1./(1+exp(-phimat * coef));  
  Q = 1 - P;
  
end

Wfd = putcoef(Wfd, coef);

%  --------------------  end of EM algorithm ---------------------
% W(theta)
plot(Wfd)
xlabel('\fontsize{16} \theta')
ylabel('\fontsize{16} W(\theta)')

% P(theta)
itemindex = 1:nit;
for j = itemindex
plot(thetaq, (CN(j,:)./N)', 'o', ...
thetaq, P(:,j), 'b-', ...
thetaq, P0(:,j), 'g--')
axis([thetamin, thetamax, 0, 1])
xlabel('\fontsize{16} \theta')
ylabel('\fontsize{16} P(\theta)')
title(['\fontsize{16} Item ',num2str(j)])
pause
end

% pdf of theta[EM算法默认theta为正态分布，此处做验证]
pdf = wgtq'.*sum(CP);
pdf = pdf./sum(pdf');
plot(thetaq,pdf)
xlabel('\fontsize{16} \theta')
ylabel('\fontsize{16} Density')
title('\fontsize{16} Probability Density Function for Trait Score Values')

% 1,9,15题立体图形
plot3(P(:,1), P(:,9), P(:,59), 'o-')
axis([0,1,0,1,0,1])
xlabel('\fontsize{16} Item 1')
ylabel('\fontsize{16} Item 9')
zlabel('\fontsize{16} Item 59')
grid on

% 1,9,59平面图
index3 = [1,9,59]
P1 = P(:,index3(1));
P2 = P(:,index3(2));
P3 = P(:,index3(3));
W1 = log(P1./(1-P1));
W2 = log(P2./(1-P2));
W3 = log(P3./(1-P3));

subplot(1,2,1)
plot(thetaq, P1, '-', thetaq, P2, '.-', thetaq, P3, 'o-')
axis([thetamin, thetamax, 0, 1])
axis('square')
xlabel('\fontsize{16} Ability \theta')
title('\fontsize{16} Probability of Success P(\theta)')

subplot(1,2,2)
plot(thetaq, W1, '-', thetaq, W2, '.-', thetaq, W3, 'o-')
axis('square')
xlabel('\fontsize{16} Ability \theta')
title('\fontsize{16} Log Odds-Ratio W(\theta)')

%--------------------------弧长参数-----------------------------------
DWfdmat = eval_fd(Wfd,thetaq,1);  % P的1阶倒数
DPmat = P.*(1-P).*DWfdmat; 
DPsqr = sum((DPmat').^2);
DPnorm = sqrt(DPsqr)';

Smax  = (thetaq(2) - thetaq(1)).*(sum(DPnorm) - ...
         0.5.*(DPnorm(1) + DPnorm(nq)));               % 弧长总长度

%Msfd = data2fd(log(DPnorm), thetaq, wbasis);
%Svec = monfn(thetaq,Msfd);
%Svec = Smax.*Svec./max(Svec);

%plot(thetaq, Svec)
%xlabel('\fontsize{16} \theta')
%ylabel('\fontsize{16} Arc Length s')

Wmat = eval_fd(Wfd, thetaq);
Wsbasis = create_bspline_basis([0,Smax], 16);
%Wsfd = data2fd(Wmat, Svec, Wsbasis);

DsPmat = DPmat./(DPnorm*ones(1,nit));

%--------------------------------PCA---------------------------------
nharm = 4;
Wpcastr = pca_fd(Wfd, nharm);
Wpcastr = varmx_pca(Wpcastr);

% 主成分在theta点处的W值
Wharmmat = eval_fd(Wpcastr.harmfd, thetaq);

% 平均的W和P
Wfdmean = mean(Wfd);
Wmean = eval_fd(Wfdmean, thetaq);
Wmeanmat = Wmean*ones(1,4);
Pmean = exp(Wmean)./(1 + exp(Wmean));

% 加减标准差并作图
Wconst = ones(nq,1)*[2, 1, .5, .5];
Wmatp = Wmeanmat + Wconst.*Wharmmat;
Wmatm = Wmeanmat - Wconst.*Wharmmat;
Pharp = exp(Wmatp)./(1+exp(Wmatp));
Pharm = exp(Wmatm)./(1+exp(Wmatm));

titlestr = ['  I: 25%'; ' II: 15%'; 'III: 24%'; ' IV: 35%'];
for j=1:nharm
subplot(2,2,j)
plot(thetaq, Pmean, '--')
axis([thetamin, thetamax, 0, 1])
text(thetaq-.1, Pharp(:,j), '+')
text(thetaq-.1, Pharm(:,j), '-')
title(['\fontsize{12} Harmonic ',titlestr(j,:)])
end
