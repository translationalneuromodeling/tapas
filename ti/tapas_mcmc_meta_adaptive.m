function [state] = tapas_mcmc_meta_adaptive(data, model, inference, state, ...
    states, si)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%


if state.nsample > inference.nburnin || mod(si, inference.ndiag)
    return;
end

t = floor(state.nsample / inference.ndiag);
t = 3;
node = 2;
c0 = 1.0;
c1 = 0.8;

% Optimal log rejection rate
ropt = 0.234;

gammaS = t^-c1;
gammas = c0*gammaS;

ns = inference.ndiag;
[np, nc] = size(state.kernel{node});
nt = size(inference.kernel{node}.k, 1);

nk = state.kernel{node};

if isfield(inference.kernel{node}, 'nuk')
    nup = inference.kernel{node}.nuk;
else
    nup = eye(size(inference.kernel{node}.k, 1));
end

samples = zeros(nt, ns - 1);

v = states{si}.v;
for j = 1 : ns
    v = v + states{si - inference.ndiag + j}.v;
end
ar = v/inference.ndiag;

for i = 1:numel(nk)
    % From Cholesky form to covariance form
    if ar(i) < 0.02
        nk{i}.s = nk{i}.s/1.5;
        continue;
    end
    k = nk{i}.k' * nk{i}.k;
    old_eig = eigs(k, 1);
    % Empirical variance
    for j = 1:ns
        samples(:, j) = states{si - j + 1}.graph{node}{i};
    end
    ek = cov(samples');
    % Set new kernel
    tk = k + gammaS * nup * ( ek - k ) * nup;
    new_eig = eigs(tk, 1);
    % Compute the Cholesky decomposition 
    try
        nk{i}.k = sparse(chol(tk));
    catch
        warning('Cholesky decomposition failed.')
        nk{i}.s = nk{k}.s/2;
        continue
    end
    % Set new scaling
    nk{i}.s = exp(log(nk{i}.s * old_eig) + gammas * (ar(i) - ropt))/new_eig;
end

state.kernel{node} = nk;

end

