function [nk] = tapas_mpdcm_fmri_update_kernel(t, ok, os, llh, ...
    ar, ptheta)
%% Computes a new kernel or covariance for the proposal distribution.
%
% ok -- Old kernel
% os -- Old samples
% ar -- Acceptance rate
%
% See Exploring an adaptative Metropolis Algorithm
% 

c0 = 1.0;
c1 = 0.8;

gammaS = t^-c1;
gammas = c0*gammaS ; 

ns = size(os, 3);
nd = size(os, 1);
nk = ok;

% Optimal log rejection rate
ropt = 0.234;

mhp = ptheta.mhp;
nmhp = sum(mhp);

for i = 1:numel(ok)
    % From Cholesky form to covariance form
    ok(i).S = ok(i).S' * ok(i).S;
    % Empirical covariance
    ts = squeeze(os(mhp, i, :));
    ts = bsxfun(@minus, ts, mean(ts, 2));
    ek = (ts * ts')./(ns-1);

    %ek = ek/eigs(ek, 1);
    % Set new kernel
    nk(i).S = ok(i).S + gammaS * ( ek - ok(i).S);
    % Compute the Cholesky decomposition
    if ar(i) < 0.01
        nk(i).S = chol(ok(i).S);
        nk(i).s = nk(i).s/1.5;
        continue
    end
    try
        nk(i).S = sparse(chol(nk(i).S));
    catch
        nk(i).S = chol(ok(i).S);
        nk(i).s = nk(i).s/1.5;
        continue
    end 
    % Set new scaling
    nk(i).s = exp(log(ok(i).s) + gammas * (ar(i) -  ropt));
end
end

