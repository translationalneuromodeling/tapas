function [R,neff,Vh,W,B,tau,thin] = psrf(varargin)
%PSRF Potential Scale Reduction Factor
%
%   [R,NEFF,V,W,B,TAU,THIN] = PSRF(X) or
%   [R,NEFF,V,W,B,TAU,THIN] = PSRF(x1,x2,...,xs)
%   returns "Potential Scale Reduction Factor" (PSRF) for collection
%   of MCMC-simulations. X is a NxDxM matrix which contains M MCMC
%   simulations of length N, each with dimension D. MCMC-simulations
%   can be given as separate arguments x1,x2,... which should have the
%   same length.
%
%   Returns 
%     R     PSRF (R=sqrt(V/W)) in row vector of length D
%     neff  estimated effective number of samples M*N/(1+2*sum(rhohat))
%     V     estimated mixture-of-sequences variances
%     W     estimated within sequence variances
%     B     estimated between sequence variances
%     TAU   estimated autocorrelation time
%     THIN  Geyer's initial positive sequence lag (useful for thinning)
%
%   The idea of the PSRF is that if R is not close to 1 (below 1.1 for
%   example) one may conclude that the tested samples were not from
%   the same distribution (chain might not have been converged yet).
%
%   Original method:
%      Brooks, S.P. and Gelman, A. (1998) General methods for
%      monitoring convergence of iterative simulations. Journal of
%      Computational and Graphical Statistics. 7, 434-455. 
%   Current version:
%      Split chains, return square-root definiton of R, and compute
%      n_eff using variogram estimate and Geyer's initial positive
%      sequence as described in Gelman et al (2013), Bayesian Data
%      Analsyis, 3rd ed, sections 11.4-11.5.
%
%   See also
%     CPSRF, MPSRF, IPSRF

% Copyright (C) 1999 Simo Särkkä
% Copyright (C) 2003-2004,2013 Aki Vehtari
%
% This software is distributed under the GNU General Public 
% Licence (version 3 or later); please refer to the file 
% Licence.txt, included with the software, for details.

% 2004-01-22 Aki.Vehtari@hut.fi Added neff, R^2->R, and cleaning
% 2013-10-20 Aki.Vehtari@aalto.fi Updated according to BDA3

X=cat(3,varargin{:});
mid=floor(size(X,1)/2);
X=cat(3,X(1:mid,:,:),X((end-mid+1):end,:,:));

[N,D,M]=size(X);

if N<=2
  error('Too few samples');
end

% Calculate means W of the variances
W = zeros(1,D);
for mi=1:M
  x = bsxfun(@minus,X(:,:,mi),mean(X(:,:,mi)));
  W = W + sum(x.*x);
end
W = W / ((N-1) * M);

% Calculate variances B (in fact B/n) of the means.
Bpn = zeros(1,D);
m = mean(reshape(mean(X),D,M)');
for mi=1:M
  x = mean(X(:,:,mi)) - m;
  Bpn = Bpn + x.*x;
end
Bpn = Bpn / (M-1);

% Calculate reduction factors
B = Bpn*N;
Vh = (N-1)/N*W + Bpn;
R = sqrt(Vh./W);  

if nargout>1
  % compute autocorrelation
  for t=1:N-1
    % variogram
    Vt(t,:)=sum(sum((X(1:end-t,:,:)-X(1+t:end,:,:)).^2,1),3)/M/(N-t);
  end
  % autocorrelation
  rho=1-bsxfun(@rdivide,Vt./2,Vh);
  % add zero lag autocorrelation
  rho=[ones(1,D);rho];

  mid=floor(N/2);
  neff=zeros(1,D);
  for di=1:D
    cp=sum(reshape(rho(1:2*mid,di),2,mid),1);
    ci=find(cp<0,1);
    if isempty(ci)
      warning(sprintf('Inital positive could not be found for variable %d, using maxlag value',di));
      ci=mid;
    else
      ci=ci-1; % last positive
    end
    cp=[cp(1:ci) 0];   % initial positive sequence
    tau(di)=-1+2*sum(cp); % initial positive sequence estimator
    neff(di)=M*N/tau(di); % initial positive sequence estimator for neff
    thin(di)=ci*2;
  end
end
