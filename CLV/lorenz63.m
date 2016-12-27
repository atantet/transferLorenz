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
tau=0.001; %time step

%
% general parameters
%

jave=0.001; % averaging length (in time units) (must be greater equal tau)
n=3; %dimension of model

total_len=10^3; % total length of trajectory (in time units)

total_steps=floor(total_len/tau); % total length of trajectory in steps
jave_steps=floor(jave/tau); % averaging length in steps
fig=0;

%
% Initial point
%
'Init'
x(1)=1;
x(2)=-0.1;
x(3)=0.1;
# Euler integration
for i=1:10^6
    xold=x;
    x(1)=xold(1)+tau*(-sigma*(xold(1)-xold(2)));
    x(2)=xold(2)+tau*(xold(1)*(r-xold(3))-xold(2));
    x(3)=xold(3)+tau*(xold(1)*xold(2)-bp*xold(3));
end
%
% Preallocation of memory
%

jlen=floor(total_len/jave); # Averaged length
start=.5*jlen; % start point for quick output (steps)
ende=start+0.05*jlen; % end point for quick output (steps)
propagator=zeros(jlen,n,n); # Propagator matrix time series
trajectory=zeros(total_steps,n); # Trajectory time series
blv=zeros(jlen,n,n); # Backward Lyapunov Vectors
tri=zeros(jlen,n,n);
lyapunov=(1:n)*0; # Lyapunov exponents
llesum=zeros(jlen,1);
cllesum=zeros(jlen,1);
%
% Initialize random set of orthogonal vectors
%
matrix=rand(n);
[matrix,R] = qr(matrix);
locjac=eye(n);

for i=1:total_steps
				%
				% Compute Propagator and trajectory.
				%

				% Per averaged step	
  if mod(i,jave_steps)==0
				# Calculate propagator
    propagator(i/jave_steps,:,:)=(eye(n)+tau*[-sigma sigma 0 ; ...
					      -x(3)+r -1 -x(1) ;...
					      x(2) x(1) -bp ])*locjac;
    locjac=eye(n);
 %
% perform qr decomposition and store backward Lyapunov Vectors and the
% upper triangular matrix
%
    
    matrix=squeeze(propagator(i/jave_steps,:,:))*matrix;
    [matrix,R] = qr(matrix);
    blv(i/jave_steps,:,:)=matrix;
    tri(i/jave_steps,:,:)=R;
				%
				% Compute Lyapunov Exponents
				%
    
    llesum(i/jave_steps)=sum(log(abs(diag(R))'))/(jave_steps*tau);
    lle(i/jave_steps,:)=log(abs(diag(R))')/(jave_steps*tau);
    
    lyapunov=lyapunov+log(abs(diag(R))');
    
  else
    locjac=(eye(n)+tau*[-sigma sigma 0 ; ...
			-x(3)+r -1 -x(1) ;...
			x(2) x(1) -bp ])*locjac;
  end
  
				%
				% Iterate to next point.
				%
  trajectory(i,:)=x;
  xold=x;
  x(1)=xold(1)+tau*(-sigma*(xold(1)-xold(2)));
  x(2)=xold(2)+tau*(xold(1)*(r-xold(3))-xold(2));
  x(3)=xold(3)+tau*(xold(1)*xold(2)-bp*xold(3));
  
  
  
  if mod(i/total_steps*1000,10)==0
    i/total_steps
  end
end
lyapunov=lyapunov/total_len;

%
% Kaplan -Yorke Dimension
%

dky=0;
i=1;
while dky >= 0
    dky=dky +lyapunov(i);
    i=i+1;
end
dky=dky-lyapunov(i-1);
dky=i-2+dky/abs(lyapunov(i-1));


%
% initialization of random upper triangular matrix and preallocation
%

clv=zeros(jlen,n,n);
matrix=rand(n);
for i=1:n
    matrix(i+1:n,i)=0;
    matrix(:,i)=matrix(:,i)/norm(matrix(:,i));
end

clle=zeros(jlen,n);
for j=jlen:-1:2
    
    matrix=(squeeze(tri(j,:,:)))^(-1)*matrix;
    for i=1:n
        factor=norm(matrix(:,i));
        clle(j,i)=clle(j,i)+log(1/factor)/(jave_steps*tau);
        matrix(:,i)=matrix(:,i)/factor;
    end
    clv(j,:,:)=squeeze(blv(j-1,:,:))*matrix;
    
    
    if mod(j/jlen*1000,10)==0
        j/jlen
    end
    
end

%
% Saving all relevant data
%
% 
% save([name,'_data'],'clv','tri','blv','trajectory','jave','total_len','total_steps','jave_steps')
% 
% 
% start=2500;
% ende=7500;
% scale=0.01;
% tra=squeeze((jave_steps*(start-1)+1:jave_steps:jave_steps*(ende-1)+1));
% figure(fig+1); fig=fig+1;
% plot3(trajectory(tra,1),trajectory(tra,2),trajectory(tra,3),'Marker','.','markersize',4,'linestyle','none','linewidth',0.1)
% hold on;
% quiver3(trajectory(tra,1),trajectory(tra,2),trajectory(tra,3),clv(start:ende,1,1),clv(start:ende,2,1),clv(start:ende,3,1),scale,'r')
% hold on;
% quiver3(trajectory(tra,1),trajectory(tra,2),trajectory(tra,3),clv(start:ende,1,2),clv(start:ende,2,2),clv(start:ende,3,2),scale,'b')
% hold on;
% quiver3(trajectory(tra,1),trajectory(tra,2),trajectory(tra,3),clv(start:ende,1,3),clv(start:ende,2,3),clv(start:ende,3,3),scale,'k')
% title({['Attractor, Lyapunov coefficients: ',num2str(lyapunov(1)),' ',num2str(lyapunov(2)),' ',num2str(lyapunov(3))] , ...
%  ['Kaplan Yorke Dimension: ',num2str(dky)]});


%% Below here are just some plots that I made at one point. Ignore if not interested

start=45000
ende=55000;
figure;
condition=clle(:,1)-clle(:,3);
cc=parula(30);
[N,edges,bin]=histcounts(condition,30);
subplot(2,2,1);
for i=start:1:ende
    (i-start+1)/(ende-start+1)
plot3(trajectory(i,1),trajectory(i,2),trajectory(i,3),'linestyle','none','marker','.','color',squeeze(cc(bin(i-start+1),:)));
if i==start; hold on;end
end
plot3(trajectory(start:1:ende,1),trajectory(start:1:ende,2),trajectory(start:1:ende,3),'color','black','linestyle','-')
subplot(2,2,3);
X=diff(edges)/2+edges(1:end-1);
for i=1:size(N,2)
bar(X(i),N(i),'facecolor',squeeze(cc(i,:)),'barwidth',edges(2)-edges(1),'edgecolor','none')
if i==1; hold on;end
end

condition=abs(squeeze(sum(clv(start:ende,:,1).*clv(start:ende,:,3),2)));
cc=parula(30);
[N,edges,bin]=histcounts(condition,30);
subplot(2,2,2);
for i=start:1:ende
    (i-start+1)/(ende-start+1)
plot3(trajectory(i,1),trajectory(i,2),trajectory(i,3),'linestyle','none','marker','.','color',squeeze(cc(bin(i-start+1),:)));
if i==start; hold on;end
end
plot3(trajectory(start:1:ende,1),trajectory(start:1:ende,2),trajectory(start:1:ende,3),'color','black','linestyle','-')
subplot(2,2,4);
X=diff(edges)/2+edges(1:end-1);
for i=1:size(N,2)
bar(X(i),N(i),'facecolor',squeeze(cc(i,:)),'edgecolor','none','barwidth',edges(2)-edges(1))
if i==1; hold on;end
end

start=20000
ende=80000
figure;
j=0;
for I=[[1 3]' [1 2]' [2 3]']
    j=j+1;
    subplot(2,2,j);
    scatter((sum(clv(start:ende,:,I(1)).*clv(start:ende,:,I(2)),2)),clle(start:ende,I(1))-clle(start:ende,I(2)),'.');
    title(num2str(I'))
end

start=20000
ende=80000
figure;
j=0;
for I=[[1 3]' [1 2]' [2 3]']
    j=j+1;
    subplot(2,2,j);
    scatter(sqrt(sum((clv(start:ende,:,I(1))-clv(start:ende,:,I(2))).^2,2)),abs(clle(start:ende,I(1))-clle(start:ende,I(2))),'.');
    title(num2str(I'))
end


i=10^3/2-1;
j=0;
for x=trajectory(10^3/2:10^3/2+100,:)'
    j=j+1
    i=i+1;
    i/10^5
    k_1 = lorenz63tend(x,para)';
    
    tendRK=(k_1)';
    tendRK=tendRK/norm(tendRK);
    short1(j)=sum(clv(i-1,:,2).*tendRK);
    short2(j)=sum(clv(i,:,2).*tendRK);
    short3(j)=sum(clv(i+1,:,2).*tendRK);
end
