function [model] = tapas_sem_hier_prepare_model(data, model, inference)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

model.graph{1}.htheta.model = ...
    tapas_sem_hier_prepare_ptheta(model.graph{1}.htheta.model);

nc = size(model.graph{2}.htheta.T, 2);
ns = numel(data);

if ~isstruct(model.graph{4}.htheta.y)
    model.graph{4}.htheta.y = struct();
end

if ~isfield(model.graph{4}.htheta.y, 'mu')
    % The prior is added after wards
    model.graph{4}.htheta.y.mu = 0 * model.graph{1}.htheta.model.njm' * ...
        model.graph{1}.htheta.model.mu;
end

np = numel(model.graph{4}.htheta.y.mu);

% Mean prior variance
mu_pe = model.graph{1}.htheta.model.njm' * model.graph{1}.htheta.model.pm;
% Assume that the variance of the variance is only max 50 
sg_pe = 0.5 * mu_pe;

[alpha, beta] = tapas_gamma_moments_to_ab(mu_pe, sg_pe);

if ~isfield(model.graph{4}.htheta.y, 'alpha')
    model.graph{4}.htheta.y.alpha =  alpha; % Wide prior
end

if ~isfield(model.graph{4}.htheta.y, 'beta')
    model.graph{4}.htheta.y.beta = beta;
end

model.graph{4}.htheta.y.eta = ones(size(mu_pe));

end
