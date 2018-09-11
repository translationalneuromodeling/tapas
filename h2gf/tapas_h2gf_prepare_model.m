function [model] = tapas_h2gf_prepare_model(data, model, inference)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

model.graph{1}.htheta.hgf = ...
    tapas_h2gf_prepare_ptheta(model.graph{1}.htheta.hgf);

if ~isstruct(model.graph{4}.htheta.y)
    model.graph{4}.htheta.y = struct();
end

if ~isfield(model.graph{4}.htheta.y, 'mu')
    model.graph{4}.htheta.y.mu = model.graph{1}.htheta.hgf.mu;
end

np = numel(model.graph{4}.htheta.y.mu);

% Mean prior variance
mu_pe = model.graph{1}.htheta.hgf.pe;
% Assume that the variance of the variance is only max 5% 
sg_pe = 0.5 * mu_pe;

[alpha, beta] = tapas_gamma_moments_to_ab(mu_pe, sg_pe);

if ~isfield(model.graph{4}.htheta.y, 'alpha')
    model.graph{4}.htheta.y.alpha =  alpha; % Wide prior
end

if ~isfield(model.graph{4}.htheta.y, 'beta')
    model.graph{4}.htheta.y.beta = beta;
end

model.graph{4}.htheta.y.eta = ...
    model.graph{1}.htheta.hgf.empirical_priors.eta;

end
