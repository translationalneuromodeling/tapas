function [inference] = tapas_sem_multiv_prepare_inference(data, model, ...
    inference)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of chains
nc = size(model.graph{2}.htheta.T, 1);
% Number of subjects
ns = numel(data);
% Number of real parameter and input parameters
[ni, np] = size(model.graph{1}.htheta.model.jm);

njm = model.graph{1}.htheta.model.njm;

% Get the kernel
if isscalar(inference.k)
    k = inference.k * eye(ni);
elseif isdiag(inference.k)
    % It is a diagonal matrix so it can be used.
    k = inference.k;
else
    % It is a vector
    k = diag(inference.k);
end
%k = blkdiag(k, k);
k = njm' * k * njm;
max_eigs = eigs(k, 1);
k = sparse(chol(k));

if isscalar(inference.k)
    no_update = logical(inference.k) * ones(1, ni);
else
    no_update = inference.k;
end

sk = inference.kernel_scale/max_eigs;

no_update = diag(no_update);
no_update = sparse(njm' * no_update * njm);

% Create a kernel for each subject and each chain.
% k is the covariance
% s is a scaling factor for the covariance
inference.kernel{2} = struct('k', k, 's', sk, 'nuk', no_update);
% No need for a kernel as it can be sampled using a gibbs step.

end

