%% Plot histograme 
% mu with three attributes
global incidenceFull; 
global Gradient;
global Op;
global Mfull;
global Ufull;
global Atts;
global Obs;     % Observation
global nbobs;  
global isLinkSizeInclusive;
global lastIndexNetworkState;
global mu;
global Scale;
global isFixedMu;
global LSatt;
global LinkSize;

Op.x = [-2.139;-0.748;-0.224;-3.301;-0.155;0.341;-0.581;-0.092];

mu = getMu(Op.x);
figure
subplot(1,2,1); % first subplot
hist(mu,100);
title('Histogram of \mu_k');
mean(mu)

% subplot(2,2,2); % first subplot
% x =  Op.x(Op.m+1: Op.n); x(3)=[];
% Sc = Scale; Sc(:,3) = [];
% mu = exp(Sc * x); 
% hist(mu,50);
% title('With TT, LF');
% mean(mu)
% 
% subplot(2,2,3); % first subplot
% x =  Op.x(Op.m+1: Op.n); x(2)=[];
% Sc = Scale; Sc(:,2) = [];
% mu = exp(Sc * x); 
% hist(mu,50);
% title('With TT, OL');
% mean(mu)
% 
% subplot(2,2,4); % first subplot
% x =  Op.x(Op.m+1: Op.n); x(1)=[];
% Sc = Scale; Sc(:,1) = [];
% mu = exp(Sc * x); 
% hist(mu,50);
% title('With LF, OL');
% mean(mu)

mu = getMu(Op.x);
%mu = randi(5,size(mu,1),size(mu,2));
%figure
Mfull = getM(Op.x, false); % matrix with exp utility for given beta
M = Mfull(1:lastIndexNetworkState,1:lastIndexNetworkState);            
M(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
M(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));      
MI = sparse(M); 
MI(find(M)) = 1;
dPhi = objArray(Op.n - Op.m);
a = mu;
k = 1 ./ mu;
 phi = sparse((k * a') .* MI);
 u  =find(phi);
 phi(u) = log(phi(u));
 subplot(1,2,2);
 hist(phi(find(phi)),100);
 title('Histogram of ln(\phi_{ka})');
mean(phi(find(phi)))
