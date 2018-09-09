function [state] = tapas_sampler_mc3(data, model, inference, state)
%% Sampler form population mcmc (mc3) 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of nodes
nn = numel(model.graph);
% Number of chains
nc = size(state.llh{1}, 2);

llh = sum(state.llh{1}, 1);
lpp = zeros(1, nc);

% Get the joint 
for i = 2:nn - 1
    lpp = lpp + sum(state.llh{i}, 1);
end

v = tapas_mc3_arc(llh, lpp, model.graph{1}.htheta.T, inference.mc3it);

for i = 1: nn - 1
    % Swap the likelihood
    state.llh{i}(:, :) = state.llh{i}(:, v);
    if i > 1
        % Swap the parameters
        state.graph{i}.y(:, :) = state.graph{i}.y(:, v); 
    end 
end

end
