close all
clear all
%
% Lorenz63
%
%

name='lorenz63';

%
% Model parameters
%

sigma=10;
r=28;
bp=8/3;
tau=0.001  %time step
sample=1;
total_len=10^3/tau/sample/2; % length of arrays in total in time dimension (mostly)

nu=0.98; % Robert-Asselin Filter

para.sigma=sigma;
para.r=r;
para.bp=bp;

jac=@(x) [-sigma sigma 0 ; ...
    -x(3)+r -1 -x(1) ;...
    x(2) x(1) -bp ]; %jacobian of Lorenz 63
L63tend=@(x,para) [(-para.sigma*(x(1)-x(2))); ...
    (x(1)*(para.r-x(3))-x(2)); ...
    (x(1)*x(2)-para.bp*x(3))]'; %Lorenz 63 Tendency
%
% general parameters
%

n=3; %dimension of model



%
% Initial point
%
'Init'
n_ens=6;
xolder=zeros(n,n_ens+1);
xold=zeros(n,n_ens+1);
x=zeros(n,n_ens+1);

xolder(1,1)=1.01;
xolder(2,1)=-0.51-0.05;
xolder(3,1)=0.01+0.044;
for i=1:10^4
    xold(:,1)=xolder(:,1)+tau/10*L63tend(xolder(:,1),para)';
    xolder(:,1)=xold(:,1);
end
matrix=rand(2*n,n_ens);
[matrix,~] = qr(matrix,0);
sample_step=1;
blv_LPF=zeros(2*n,n_ens,total_len);
tri_LPF=zeros(n_ens,n_ens,total_len);
loclyap_LPF=zeros(n_ens,total_len);
trajectory_LPRA=zeros(2*n,total_len);
for ensemble_length=[10.^(-8)];
    
    xolder(:,2:n_ens+1)=bsxfun(@plus,ensemble_length*matrix(1:n,1:n_ens),xolder(:,1));
    xold(:,2:n_ens+1)=bsxfun(@plus,ensemble_length*matrix(n+1:end,1:n_ens),xold(:,1));
    trajectory_LPRA(1:n,1)=xolder(:,1);
    trajectory_LPRA(n+1:2*n,1)=xold(:,1);
    
    for i=1:total_len
        trajectory_LPRA(1:n,i)=xolder(:,1);
        trajectory_LPRA(n+1:2*n,i)=xold(:,1);
        for ens=0:n_ens
            for s=1:sample
                x(:,ens+1)=xold(:,ens+1)*(1-nu)+nu*(xolder(:,ens+1)+tau*L63tend(xold(:,ens+1),para)');
                xolder(:,ens+1)=xold(:,ens+1);
                xold(:,ens+1)=x(:,ens+1);
                x(:,ens+1)=xold(:,ens+1)*(1-nu)+nu*(xolder(:,ens+1)+tau*L63tend(xold(:,ens+1),para)');
                xolder(:,ens+1)=xold(:,ens+1);
                xold(:,ens+1)=x(:,ens+1);
            end
        end
        matrix(1:n,:)=bsxfun(@minus,xolder(:,2:n_ens+1),xolder(:,1))/ensemble_length;
        matrix(n+1:end,:)=bsxfun(@minus,xold(:,2:n_ens+1),xold(:,1))/ensemble_length;
        [matrix,R]=qr(matrix,0);
        signumR=diag(diag(sign(R(1:n_ens,1:n_ens))),0);
        blv_LPF(:,:,i)=matrix(:,1:n_ens);
        tri_LPF(:,:,i)=R(1:n_ens,1:n_ens);
        loclyap_LPF(:,i)=log(abs(diag(tri_LPF(:,:,i))))/(tau*sample);
        
        for ens=1:n_ens
            xolder(:,ens+1)=ensemble_length*matrix(1:n,ens)+xolder(:,1);
            xold(:,ens+1)=ensemble_length*matrix(n+1:end,ens)+xold(:,1);
        end
        if mod(i-1,1000)==0
            display(['Percent ',num2str(i/total_len,'%6.4f'),'; Lyapunov 1: ',num2str(mean(loclyap_LPF(1,1:i),2)),'; Lyapunov 2: ',num2str(mean(loclyap_LPF(2,1:i),2)),'; Lyapunov 3: ',num2str(mean(loclyap_LPF(3,1:i),2))]);
            if n_ens==6; display(['Percent ',num2str(i/total_len,'%6.4f'),'; Lyapunov 4: ',num2str(mean(loclyap_LPF(4,1:i),2)),'; Lyapunov 5: ',num2str(mean(loclyap_LPF(5,1:i),2)),'; Lyapunov 6: ',num2str(mean(loclyap_LPF(6,1:i),2))]);end
        end
    end
    LLE_LPF(:,round(-log(ensemble_length)/log(10)))=mean(loclyap_LPF(:,:),2)';
end

%
% initialization of random upper triangular matrix and preallocation
%

clv_LPF=zeros(size(blv_LPF));
matrix=rand(n_ens,n_ens);
for i=1:n_ens
    matrix(i+1:end,i)=0;
    matrix(:,i)=matrix(:,i)/norm(matrix(:,i));
end

clear clv_LPF_disentangled
clle_LPF=zeros(size(loclyap_LPF));
for j=total_len/2:-1:2
    
    matrix=(squeeze(tri_LPF(:,:,j)))^(-1)*matrix;
    for i=1:n_ens
        factor=norm(matrix(:,i));
        clle_LPF(i,j)=clle_LPF(i,j)+log(1/factor)/(tau);
        matrix(:,i)=matrix(:,i)/factor;
    end
    clv_LPF(:,:,j)=squeeze(blv_LPF(:,1:n_ens,j-1))*matrix;
    clv_LPF_disentangled(:,:,2*j)=clv_LPF(n+1:end,1:n_ens,j)/norm(clv_LPF(n+1:end,1:n_ens,j));
    clv_LPF_disentangled(:,:,2*j-1)=clv_LPF(1:n,1:n_ens,j)/norm(clv_LPF(1:n,1:n_ens,j));
end

%
% Rescale Leapfrog vectors individually
%
clear blv_LPF_rescaled clv_LPF_rescaled
blv_LPF_rescaled(1:3,:,:)=bsxfun(@times,blv_LPF(1:3,:,:),1./sqrt(sum(blv_LPF(1:3,:,:).^2,1)));
blv_LPF_rescaled(4:6,:,:)=bsxfun(@times,blv_LPF(4:6,:,:),1./sqrt(sum(blv_LPF(4:6,:,:).^2,1)));

clv_LPF_rescaled(1:3,:,:)=bsxfun(@times,clv_LPF(1:3,:,:),1./sqrt(sum(clv_LPF(1:3,:,:).^2,1)));
clv_LPF_rescaled(4:6,:,:)=bsxfun(@times,clv_LPF(4:6,:,:),1./sqrt(sum(clv_LPF(4:6,:,:).^2,1)));



%
% Runge-Kutta 4th order
%
xold=zeros(n,1);
xold(1)=1.01;
xold(2)=-0.1-0.05;
xold(3)=0.1+0.044;
for i=1:10^4
    xold(:,1)=xold(:,1)+tau/10*L63tend(xold(:,1),para)';
end%
% Initialize random set of orthogonal vectors
%

matrix=rand(n);
[matrix,R] = qr(matrix);

prop=eye(n);
trajectory_RK4th = zeros(n,total_len);
tendency_RK4th = zeros(n,total_len);
propagator_RK4th= zeros(n,n,total_len);
blv_RK4th=zeros(n,n,total_len);
tri_RK4th=zeros(n,n,total_len);
loclyap_RK4th=zeros(n,total_len);

for i=1:total_len
    i/total_len
    trajectory_RK4th(:,i) = xold;
    prop=eye(n);
    for s=1:sample
        k_1 = L63tend(xold,para)';
        k_2 = L63tend(xold+0.5*tau*k_1,para)';
        k_3 = L63tend(xold+0.5*tau*k_2,para)';
        k_4 = L63tend(xold+tau*k_3,para)';
        x    = xold    + 1/6*tau*(k_1+2*k_2+2*k_3+k_4);
        
        j_1=jac(xold);
        j_2=jac(xold+0.5*tau*k_1)*(eye(n)+0.5*tau*j_1);
        j_3=jac(xold+0.5*tau*k_2)*(eye(n)+0.5*tau*j_2);
        j_4=jac(xold+tau*k_3)*(eye(n)+tau*j_3);
        prop = (eye(n) + 1/6*tau*(j_1+2*j_2+2*j_3+j_4)) * prop;
        
        xold = x;
    end
    
    matrix=prop*matrix;
    [matrix,R] = qr(matrix);
    propagator_RK4th(:,:,i)=prop;
    signumR=eye(n);%diag(diag(sign(R)),0);
    blv_RK4th(:,:,i)=matrix*signumR;
    tri_RK4th(:,:,i)=signumR*R;
    loclyap_RK4th(:,i)=log(abs(diag(signumR*R)))/(tau*sample);
    
end

%
% initialization of random upper triangular matrix and preallocation
%

clv_RK4th=zeros(size(blv_RK4th));
matrix=rand(n);
for i=1:n
    matrix(i+1:n,i)=0;
    matrix(:,i)=matrix(:,i)/norm(matrix(:,i));
end

clle_RK4th=zeros(size(loclyap_RK4th));
clv_RK4th=zeros(size(blv_RK4th));
correlationtend1=zeros(1,total_len);
correlationtend2=zeros(1,total_len);
correlationtend3=zeros(1,total_len);
for j=total_len:-1:2
    j/total_len
    matrix=(squeeze(tri_RK4th(:,:,j)))^(-1)*matrix;
    for i=1:n
        factor=norm(matrix(:,i));
        clle_RK4th(i,j)=clle_RK4th(i,j)+log(1/factor)/(tau*sample);
        matrix(:,i)=matrix(:,i)/factor;
    end
    clv_RK4th(:,:,j)=squeeze(blv_RK4th(:,:,j-1))*matrix;
    xold=trajectory_RK4th(:,j);
    k_1 = L63tend(xold,para)';
    k_2 = L63tend(xold+0.5*tau*k_1,para)';
    k_3 = L63tend(xold+0.5*tau*k_2,para)';
    k_4 = L63tend(xold+tau*k_3,para)';
    tendRK=(k_1+2*k_2+2*k_3+k_4);
    tendRK=tendRK/norm(tendRK);
    tendency(:,j)=tendRK;
    correlationtend1(j)=sum(clv_RK4th(:,1,j).*tendRK);
    correlationtend2(j)=sum(clv_RK4th(:,2,j).*tendRK);
    correlationtend3(j)=sum(clv_RK4th(:,3,j).*tendRK);
end

save('leapfrog','-v7.3')

A=5000;
B=45000;
dim=2;
figure;
subplot(2,3,1)
histogram(-[blv_LPF_rescaled(dim,1,A:B) blv_LPF_rescaled(dim+3,1,A:B)],'edgecolor','none','normalization','probability');
hold on;
histogram(blv_RK4th(dim,1,A:B),50,'edgecolor','none','normalization','probability')
title('BLV 1')
legend({'LeapFrog' 'Runge Kutta 4th'})
subplot(2,3,2)
histogram(-[blv_LPF_rescaled(dim,2,A:B) blv_LPF_rescaled(dim+3,2,A:B)],50,'edgecolor','none','normalization','probability');
hold on;
histogram(blv_RK4th(dim,2,A:B),50,'edgecolor','none','normalization','probability')
title('BLV 2')
legend({'LeapFrog' 'Runge Kutta 4th'})
subplot(2,3,3)
histogram(-[blv_LPF_rescaled(dim,3,A:B) blv_LPF_rescaled(dim+3,3,A:B)],50,'edgecolor','none','normalization','probability');
hold on;
histogram(blv_RK4th(dim,3,A:B),50,'edgecolor','none','normalization','probability')
title('BLV 3')
legend({'LeapFrog' 'Runge Kutta 4th'})
% subplot(2,3,4)
% histogram([blv_LPF_rescaled(dim,4,A:B) blv_LPF_rescaled(dim+3,4,A:B)],'edgecolor','none','normalization','probability');
% title('BLV 4')
% subplot(2,3,5)
% histogram([blv_LPF_rescaled(dim,5,A:B) blv_LPF_rescaled(dim+3,5,A:B)],'edgecolor','none','normalization','probability');
% title('BLV 5')
% subplot(2,3,6)
% histogram([blv_LPF_rescaled(dim,6,A:B) blv_LPF_rescaled(dim+3,6,A:B)],'edgecolor','none','normalization','probability');
% title('BLV 6')

subplot(2,3,4)
histogram(-[clv_LPF_rescaled(dim,1,A:B) clv_LPF_rescaled(dim+3,1,A:B)],50,'edgecolor','none','normalization','probability');
hold on;
histogram(clv_RK4th(dim,1,A:B),50,'edgecolor','none','normalization','probability')
title('clv 1')
legend({'LeapFrog' 'Runge Kutta 4th'})
subplot(2,3,5)
histogram(-[clv_LPF_rescaled(dim,2,A:B) clv_LPF_rescaled(dim+3,2,A:B)],'edgecolor','none','normalization','probability');
hold on;
histogram(clv_RK4th(dim,2,A:B),50,'edgecolor','none','normalization','probability')
title('clv 2')
legend({'LeapFrog' 'Runge Kutta 4th'})
subplot(2,3,6)
histogram(-[clv_LPF_rescaled(dim,3,A:B) clv_LPF_rescaled(dim+3,3,A:B)],50,'edgecolor','none','normalization','probability');
hold on;
histogram(clv_RK4th(dim,3,A:B),50,'edgecolor','none','normalization','probability')
title('clv 3')
legend({'LeapFrog' 'Runge Kutta 4th'})
% subplot(2,3,4)
% histogram([clv_LPF_rescaled(dim,4,A:B) clv_LPF_rescaled(dim+3,4,A:B)],'edgecolor','none','normalization','probability');
% title('clv 4')
% subplot(2,3,5)
% histogram([clv_LPF_rescaled(dim,5,A:B) clv_LPF_rescaled(dim+3,5,A:B)],'edgecolor','none','normalization','probability');
% title('clv 5')
% subplot(2,3,6)
% histogram([clv_LPF_rescaled(dim,6,A:B) clv_LPF_rescaled(dim+3,6,A:B)],'edgecolor','none','normalization','probability');
% title('clv 6')


full=reshape([trajectory_LPRA(1:3,:) ;trajectory_LPRA(4:6,:)],3,[]);

figure;
cc={'r' 'b' ,'k'};
RKtra=(10100:10110);
RKtra2=(2*RKtra(1):2*RKtra(end));
for clv=1:3
    S=1;
    quiver3(trajectory_LPRA(1,RKtra(1:S:end))',trajectory_LPRA(2,RKtra(1:S:end))',trajectory_LPRA(3,RKtra(1:S:end))',squeeze(clv_LPF(1,clv,RKtra(1:S:end))),squeeze(clv_LPF(2,clv,RKtra(1:S:end))),squeeze(clv_LPF(3,clv,RKtra(1:S:end))),0.1,cc{clv})
    hold on;
    quiver3(trajectory_LPRA(4,RKtra(1:S:end))',trajectory_LPRA(5,RKtra(1:S:end))',trajectory_LPRA(6,RKtra(1:S:end))',squeeze(clv_LPF(3+1,clv,RKtra(1:S:end))),squeeze(clv_LPF(3+2,clv,RKtra(1:S:end))),squeeze(clv_LPF(3+3,clv,RKtra(1:S:end))),0.1,cc{clv})
    plot3(trajectory_LPRA(1,RKtra),trajectory_LPRA(2,RKtra),trajectory_LPRA(3,RKtra),'linestyle','none','color','black','marker','+');
    plot3(trajectory_LPRA(4,RKtra),trajectory_LPRA(5,RKtra),trajectory_LPRA(6,RKtra),'linestyle','none','color','black','marker','o');
    plot3(full(1,RKtra2),full(2,RKtra2),full(3,RKtra2),'-')
end


figure;
cc={'r' 'b' ,'k' 'y' 'g' ,'m'};
RKtra=(15000:15100);
RKtra2=(2*RKtra(1):2*RKtra(end));
for clv=4:6
    S=1;
    quiver3(trajectory_LPRA(1,RKtra(1:S:end))',trajectory_LPRA(2,RKtra(1:S:end))',trajectory_LPRA(3,RKtra(1:S:end))',squeeze(clv_LPF(1,clv,RKtra(1:S:end))),squeeze(clv_LPF(2,clv,RKtra(1:S:end))),squeeze(clv_LPF(3,clv,RKtra(1:S:end))),0.1,cc{clv})
    hold on;
    quiver3(trajectory_LPRA(4,RKtra(1:S:end))',trajectory_LPRA(5,RKtra(1:S:end))',trajectory_LPRA(6,RKtra(1:S:end))',squeeze(clv_LPF(3+1,clv,RKtra(1:S:end))),squeeze(clv_LPF(3+2,clv,RKtra(1:S:end))),squeeze(clv_LPF(3+3,clv,RKtra(1:S:end))),0.1,cc{clv})
    plot3(trajectory_LPRA(1,RKtra),trajectory_LPRA(2,RKtra),trajectory_LPRA(3,RKtra),'linestyle','none','color','black','marker','+');
    plot3(trajectory_LPRA(4,RKtra),trajectory_LPRA(5,RKtra),trajectory_LPRA(6,RKtra),'linestyle','none','color','black','marker','o');
    plot3(full(1,RKtra2),full(2,RKtra2),full(3,RKtra2),'-')
end


RKtra=(2000:2910);
RKtra2=(2*RKtra(1):2*RKtra(end));
figure;
cc={'r' 'b' ,'k'};
for clv=1:3
    quiver3(trajectory_RK4th(1,RKtra2)',trajectory_RK4th(2,RKtra2)',trajectory_RK4th(3,RKtra2)', ...
        squeeze(clv_RK4th(1,clv,RKtra2)),squeeze(clv_RK4th(2,clv,RKtra2)),squeeze(clv_RK4th(3,clv,RKtra2)),0.1,cc{clv})
    hold on;
end
plot3(trajectory_RK4th(1,RKtra2)',trajectory_RK4th(2,RKtra2)',trajectory_RK4th(3,RKtra2)','linewidth',0.1,'color','black');

