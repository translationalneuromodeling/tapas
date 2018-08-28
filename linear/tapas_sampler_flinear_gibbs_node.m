function [nstate] = tapas_sampler_flinear_gibbs_node(data, model, ...
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

[np, nc] = size(state.graph{node - 1});
nm = numel(state.graph{node}.y{1}.mu);
% Compute the means

mu = cell(1, nc);

% Initilize at the prior
y = state.graph{node + 1}.y;

for i = 1:nc
    for j = 1:np
        y{i}.mu = y{i}.mu + state.graph{node - 1}.y{j, i};
    end
    y{i}.mu = y{i}.mu/(np + 1);
end

% Compute the variance

% var(x) = E[x**2] - (E[x]**2)

vt = zeros(1, nc);

for i = 1:nc
    vt(1, i) = sum(state.graph{node + 1}.y{i}.mu .^ 2);
    for j = 1:np
        vt(1, i) = vt(1, i) + sum(state.graph{node - 1}.y{j, i} .^ 2);
    end
    vt(1, i) = vt(1, i) - sum(y{i}.mu .^ 2);
end

k = zeros(1, nc);
t = zeros(1, nc);

for i = 1:nc
    k = state.graph{node + 1}.y{i}.k;
    t = state.graph{node + 1}.y{i}.t;
end

% Sample the variance
pe = gamrnd(k, 1 ./ t);

for i = 1:nc
    nstate.graph{node}.y{i}.pe = pe(i);
    nstate.graph{node}.y{i}.mu = y{i}.mu + 1/pe(i) * rand(nm, 1);
end

nstate.llh{node} = model.graph{node}.llh(nstate.graph{node}, ...
    nstate.graph{node + 1}, model.graph{node}.htheta);

nstate.llh{node - 1} = model.graph{node - 1}.llh(...
    nstate.graph{node - 1}, nstate.graph{node}, model.graph{node - 1}.htheta);

end

