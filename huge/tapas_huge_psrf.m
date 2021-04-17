function [ psrf, neff, tau, thin ] = tapas_huge_psrf( samples, nChains, rBurnIn )
% Calculate the Potential Scale Reduction Factor
%
% INPUTS:
%   samples - array containing samples from MCMC. 1st dimension: samples,
%             2nd dimension: parameters, 3rd dimension: chains.
% 
% OPTIONAL INPUTS:
%   nChains - number of chains. If supplied 3rd dimension of samples is
%             interpreted as parameters.
%   rBurnIn - ratio of samples to discard for burn-in.
% 
% OUTPUTS:
%   psrf - array containing Potential Scale Reduction Factor
%   neff - effective sample size
%   tau  - autocorrelation time
%   thin - length of initial positive sequence
% 

% REFERENCE:
%   Stephen P. Brooks & Andrew Gelman (1998) General Methods for
%   Monitoring Convergence of Iterative Simulations, Journal of
%   Computational and Graphical Statistics, 7:4, 434-455, DOI:
%   10.1080/10618600.1998.10474787

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2020 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


[N, P, M] = size(samples);
if nargin < 2
    M = 1;
end
if nargin < 3
    rBurnIn = 0; % ratio of samples to discard for burn-in
end
psrf = zeros(P, M);
if nargout > 1
    neff = zeros(P, M);
    tau  = zeros(P, M);
    thin = zeros(P, M);
end

for p = 1:P
    for m = 1:M
        if nargin < 2 % treat 3rd dimension as chains
            tmp = squeeze(samples(:, p, :));         
        else % subdivide 1st dimension into pseudo-chains
            tmp = reshape(samples(end-fix(N/nChains)*nChains+1:end, p, m), ...
                fix(N/nChains), nChains);
        end
        psrf(p, m) = calculate_psrf( tmp, rBurnIn );
        if nargout > 1
            [ neff(p, m), tau(p, m), thin(p, m) ] = calculate_neff( tmp, rBurnIn );
        end
    end
end

end

function [ psrf ] = calculate_psrf( samples, bi )

[N, M] = size(samples);
assert(M > 1, 'TAPAS:HUGE:noChains', 'Not enough chains.')

% discard first part of each chain as burn-in
samples = samples(max(1, fix(N*bi)):end, :);
N = size(samples, 1);
assert(N > 9, 'TAPAS:HUGE:noSamples', 'Not enough samples.')

% within-chain mean and variance
x_bar = mean(samples);
s2 = var(samples);

B = var(x_bar);
W = mean(s2);

% across-chain mean and variance
sigma_hat2 = (N - 1)/N*W + B;

V_hat = sigma_hat2 + B/M;
psrf = sqrt(V_hat/W);

% % correction based on t-distribution
% mu_hat = mean(samples(:));
% var_V_hat = (N - 1)^2/N^2/M*var(s2) + 2*(M + 1)^2/M^2/(M-1)*B^2 ...
%     + 2*(M + 1)*(N - 1)/M^2/N* ...
%     (xcov(s2, x_bar.^2, 0) - 2*mu_hat*xcov(s2, x_bar, 0));
% df = 2*V_hat^2/var_V_hat;
% psrf = sqrt(V_hat/W*(df + 3)/(df + 1));

end

function [ neff, tau, thin ] = calculate_neff( samples, bi )

% discard first part of each chain as burn-in
[N,M] = size(samples);
samples = samples(max(1, fix(N*bi)):end, :);
N = size(samples, 1);
K = 2*floor(N/2) - 1;
% autocovariance
rho = zeros(2*N-1,1);
for m = 1:M
  rho = rho + xcov(samples(:,m),'coeff');
end
thin = 2*(find(rho(N:2:N+K)+rho(N+1:2:N+K)<0,1,'first') - 1) - 1;
if isempty(thin)
    thin = K;
end
thin = max(thin,1);
tau = sum(rho(N-thin:N+thin))/M; % Autocorrelation time estimation
neff = M*N/tau; % effective sample size

end
