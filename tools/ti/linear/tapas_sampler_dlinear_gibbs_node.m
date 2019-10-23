function [nstate] = tapas_sampler_dlinear_gibbs_node(data, model, ...
    inference, state, node)
%% Samples from a linear model with variance using a gibbs step.
%
% Input 
%
% Output
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nstate = state;

[np, nc] = size(state.graph{node - 1}.y);
nm = numel(state.graph{node}.y{1}.mu);
% Compute the means

mu = cell(1, nc);

% Initilize at the prior
y = state.graph{node + 1}.y;

% First store the values somewhere
% np + 1 Number of subjects plus prior
values = zeros(size(y{1}.mu, 1), np);

% COMMENT: go through this.
for i = 1:nc
    alpha = y{i}.alpha;
    beta = y{i}.beta;
    mu0 = y{i}.mu;
    eta = y{i}.eta; % Overweight prior

    for j = 1:np
        values(:, j)  = state.graph{node - 1}.y{j, i};
    end
    means = (mu0 .* eta +  sum(values, 2)) ./ (np + eta);
    residuals = eta .* (mu0 - means).^2.0 + sum((mu0 - means).^2, 2);
    % Priors 
    % Sample variance for each of the components
    % The one degree of freedom comes from the prior
    pe = gamrnd(alpha + (np + eta)/2, 1./(beta + residuals/2));
    nstate.graph{node}.y{i}.pe = pe;
    % Update the mean
    nstate.graph{node}.y{i}.mu = means + ...
        (1./sqrt((np + 1) * pe) .* randn(nm, 1));
end

% COMMENT: Conditional probability of subject-specific parameters given
% the newly sampled population mean and variance. Needed for Metropolis-
% Hastings step at level below.
nstate.llh{node - 1} = model.graph{node - 1}.llh(...
    nstate.graph{node - 1}, ...
    nstate.graph{node}, ...
    model.graph{node - 1}.htheta);

end
