function [inference] = tapas_h2gf_prepare_inference(data, model, inference)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

nc = size(model.graph{2}.htheta.T, 1);
ns = numel(data);
np = numel(model.graph{4}.htheta.y.mu);

%k = cell(ns, nc);
k = eye(np);

% Create a kernel for each subject and each chain.
% k is the covariance
% s is a scaling factor for the covariance
inference.kernel{2} = struct('k', k, 's', 0.5);
% No need for a kernel as it can be sampled using a gibbs step.

end

