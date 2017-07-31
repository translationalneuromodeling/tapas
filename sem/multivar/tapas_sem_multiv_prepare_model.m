function [model] = tapas_sem_multiv_prepare_model(data, model, inference)
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
    tapas_sem_multiv_prepare_ptheta(model.graph{1}.htheta.model);

% The trick is that now njm includes also the information about the 
% regressor matrix 

% Effective number of parameters
np = size(model.graph{1}.htheta.model.njm, 2);
% Number of regressors
nr = size(model.graph{1}.htheta.model.x, 2);


nc = size(model.graph{2}.htheta.T, 2);
ns = numel(data);

if ~isstruct(model.graph{4}.htheta.y)
    model.graph{4}.htheta.y = struct();
end

if ~isfield(model.graph{4}.htheta.y, 'mu')
    % Assume rather wide prior
    model.graph{4}.htheta.y.mu = zeros(nr, np);
end

np = numel(model.graph{4}.htheta.y.mu);

% Mean prior variance
mu_pe = model.graph{1}.htheta.model.njm' * model.graph{1}.htheta.model.pm;
% Assume big variance of the variance
sg_pe = 5.0 * mu_pe;

[alpha, beta] = tapas_gamma_moments_to_ab(mu_pe, sg_pe);

if ~isfield(model.graph{4}.htheta.y, 'alpha')
    model.graph{4}.htheta.y.alpha =  alpha'; % Wide prior
end

if ~isfield(model.graph{4}.htheta.y, 'beta')
    model.graph{4}.htheta.y.beta = beta';
end


end
