function [nstate] = tapas_sampler_mdlinear_gibbs_node(data, model, ...
    inference, state, node)
%% Samples from a linear multivariate model with diagonal, parameters 
% specific covariance.
%
% Input 
%
% Output
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nstate = state;

% Number of subjects and number of chains
[ns, nc] = size(state.graph{node - 1}.y);

% Number of parameters
np = size(state.graph{node}.y{1}.mu, 2);

% Number of regressors
nr = size(state.graph{node}.y{1}.mu, 1);

% First store the values somewhere

for i = 1:nc
    % Get all the parameters at a given temperature
    iy = cell2mat(state.graph{node - 1}.y(:, i));
    iy = reshape(iy, np, ns)';
    % Now go parameter after parameter
    means = state.graph{node - 1}.u.iomega * ...
        (state.graph{node - 1}.u.x' * iy + ...
        state.graph{node + 1}.y{i}.mu);
    % One can show that the term that goes in the variance is the residuals
    % with respect to the observations and the prior
    ry = state.graph{node - 1}.u.x * means - iy;
    rm = means - state.graph{node + 1}.y{i}.mu;
    variance = sum(ry .* ry, 1) + sum(rm .* rm, 1);

    % Priors 
    alpha = state.graph{node + 1}.y{i}.alpha;
    beta = state.graph{node + 1}.y{i}.beta;
    % Sample variance for each of the components
    pe = gamrnd(alpha + ns/2, 1./(beta + variance/2));
    nstate.graph{node}.y{i}.pe = pe;
    % Update the mean
    nstate.graph{node}.y{i}.mu = means + ...
        state.graph{node - 1}.u.ciomega * bsxfun(@times, sqrt(1./pe), ...
        randn(nr, np));
end

nstate.llh{node} = model.graph{node}.llh(nstate.graph{node}, ...
    nstate.graph{node + 1}, model.graph{node}.htheta);

nstate.llh{node - 1} = model.graph{node - 1}.llh(...
    nstate.graph{node - 1}, nstate.graph{node}, model.graph{node - 1}.htheta);

end

