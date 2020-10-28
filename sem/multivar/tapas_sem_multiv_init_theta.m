function [theta] =  tapas_sem_multiv_init_theta(data, model, inference)
%% Obtain an initial sample with positive likelihood. 
%
% aponteeduardo@gmail.com
% copyright (C) 2017
%

ptheta = model.graph{1}.htheta.model;

nc = size(model.graph{1}.htheta.T, 2);
ns = size(data, 1);
mu = model.graph{4}.htheta.y.mu;
nb = numel(mu);
np = size(ptheta.jm, 2);

ty = ptheta.x * model.graph{4}.htheta.y.mu;
ty = reshape(ty', numel(ty), 1);
theta = repmat(mat2cell(ty, np * ones(ns, 1), 1), 1, 1);

llh = tapas_sem_multiv_llh(data, struct('y', {theta}), model.graph{1}.htheta);
failing = find(llh == -inf);

tolerance = 1000;
while numel(failing) && tolerance
    for i = failing'
        theta{i} = sample_gaussian(ptheta);
    end
    llh = tapas_sem_multiv_llh(data, struct('y', {theta}), ...
        model.graph{1}.htheta);
    failing = find(llh == -inf); 
    tolerance = tolerance - 1;
end

if numel(failing)
    error('tapas:sem:multiv:init', ...
    'It was not possible to initialize a sample with positive likelihood');
end

theta = repmat(theta, 1, nc);

end

function [theta] = sample_gaussian(ptheta)
%% Sample from a Gaussian prior.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

np = size(ptheta.jm, 2);

sample_pars = logical(sum(ptheta.jm, 2));

if size(ptheta.pm, 2) ~= np
    pm = diag(ptheta.pm);
else
    pm = ptheta.pm;
end

lt = chol(pm);
theta = lt \ ptheta.jm * randn(np, 1);
theta = ptheta.sm' * theta;

end

