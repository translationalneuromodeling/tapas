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

% First level: prepare vectors containing prior means and precisions
model.graph{1}.htheta.hgf = ...
    tapas_h2gf_prepare_ptheta(model.graph{1}.htheta.hgf);


% Fourth level

% Prior means from HGF Toolbox config file (preferred name: mu_0)
mu = model.graph{1}.htheta.hgf.mu;
% Prior precisions from HGF Toolbox config file (preferred name: tau_0)
pe = model.graph{1}.htheta.hgf.pe;
% Weight of prior
eta = model.graph{1}.htheta.hgf.empirical_priors.eta;

% Gaussian-gamma shape hyperparameter
% COMMENT: With this value, it implies that the Gaussian and
% gamma parts of the Gaussian-gamma hyperprior are informed by the same
% number of virtual observations. In principle, it may take any positive
% value. IT SHOULD BE CONFIGURABLE WITH THIS VALUE AS THE DEFAULT.
alpha = (eta + 1)./2;
% Gaussian-gamma rate hyperparameter
beta = eta./(2.*pe);

% Fill the y structure at the fourth level
if ~isstruct(model.graph{4}.htheta.y)
    model.graph{4}.htheta.y = struct();
end

if ~isfield(model.graph{4}.htheta.y, 'mu')
    model.graph{4}.htheta.y.mu = mu;
end

if ~isfield(model.graph{4}.htheta.y, 'alpha')
    model.graph{4}.htheta.y.alpha = alpha;
end

if ~isfield(model.graph{4}.htheta.y, 'beta')
    model.graph{4}.htheta.y.beta = beta;
end

model.graph{4}.htheta.y.eta = eta;

end
