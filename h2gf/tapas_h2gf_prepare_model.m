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
% Weight of prior
eta = model.graph{1}.htheta.hgf.empirical_priors.eta;
% Alpha parameter of gamma distribution
alpha = model.graph{1}.htheta.hgf.empirical_priors.alpha;
% Beta parameters of the gamma distribution
beta = model.graph{1}.htheta.hgf.empirical_priors.beta;


% Fill the y structure at the fourth level
if ~isstruct(model.graph{4}.htheta.y)
    model.graph{4}.htheta.y = struct();
end

% Mean, alpha and beta, and weight of the prior (eta)
model.graph{4}.htheta.y.mu = mu;
model.graph{4}.htheta.y.alpha = alpha;
model.graph{4}.htheta.y.beta = beta;
model.graph{4}.htheta.y.eta = eta;

end
