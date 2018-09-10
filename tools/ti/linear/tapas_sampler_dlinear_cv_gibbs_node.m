function [nstate] = tapas_sampler_dlinear_cv_gibbs_node(data, model, ...
    inference, state, node)
%% Samples from a linear model with variance using a gibbs step and a 
% leave one out approach.
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
values = zeros(size(y{1}.mu, 1), np + 1);
variance = cell(nc, 1);
for i = 1:nc
    values(:, 1) = y{i}.mu;
    for j = 1:np
        values(:, j + 1)  = state.graph{node - 1}.y{j, i};
    end
    y{i}.mu = mean(values, 2);
    % No bessel correction
    variance{i} = var(values, 1, 2);
end

for i = 1:nc
    % Priors 
    alpha = state.graph{node + 1}.y{i}.alpha;
    beta = state.graph{node + 1}.y{i}.beta;
    % Sample variance for each of the components
    % The one degree of freedom comes from the prior
    pe = gamrnd(alpha + np/2, 1./(beta + ((np + 1)/2.0) .* variance{i}));
    nstate.graph{node}.y{i}.pe = pe;
    % Update the mean
    nstate.graph{node}.y{i}.mu = y{i}.mu + ...
        (1./sqrt((np + 1) * pe) .* randn(nm, 1));
end

nstate.llh{node} = model.graph{node}.llh(nstate.graph{node}, ...
    nstate.graph{node + 1}, model.graph{node}.htheta);

nstate.llh{node - 1} = model.graph{node - 1}.llh(...
    nstate.graph{node - 1}, nstate.graph{node}, model.graph{node - 1}.htheta);

end

