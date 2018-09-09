function [nstate] = tapas_sampler_mixedlinear_gibbs_node(data, model, ...
    inference, state, node)
%% Samples from a linear multivariate model with fixed and random effects
% and a diagonal covariance matrix.
%
% Input 
%
% Output
%
% This level assumes only and unknown mean with known variance for some the
% the elements. The input matrix is required to be simple a weighting of 
% prior and mean. 
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
    iy = state.graph{node - 1}.y{i}.mu;
    
    % Get the grouping of the random variables.
    x = state.graph{node - 1}.u.x;
    
    % The number of groups is equal to size(x, 2). The ones in a column
    % encode membership to a code. size(x, 1) should be np;

    % Number of groups
    ng = size(x, 2); 
    
    for j = 1:ng
        % Group 
        gv = logical(x(:, j));
        ngv = sum(gv);
        % Group mean
        gmu = sum(iy(gv, :), 1) .* state.graph{node - 1}.y{i}.pe;
        gmu = gmu + state.graph{node + 1}.y{i}.pe(j, :) .* ...
            state.graph{node + 1}.y{i}.mu(j, :); % 1 x np

        % Group precision
        gpe = ngv * state.graph{node - 1}.y{i}.pe + ...
            state.graph{node + 1}.y{i}.pe(j, :);

        % Sample using Gibbs
        gmu = gmu./gpe;
        gmu = gmu + sqrt(1./ gpe) .* randn(1, np);

        % Replicate for all the correspondign regressors.
        nstate.graph{node}.y{i}.mu(gv, :) = repmat(gmu, ngv, 1);
    end
end

% Don't need it
%nstate.llh{node} = model.graph{node}.llh(nstate.graph{node}, ...
%    nstate.graph{node + 1}, model.graph{node}.htheta);
%
%nstate.llh{node - 1} = model.graph{node - 1}.llh(...
%    nstate.graph{node - 1}, nstate.graph{node}, model.graph{node - 1}.htheta);

end

