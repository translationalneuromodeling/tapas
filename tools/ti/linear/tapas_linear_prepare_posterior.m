function [posterior] = tapas_linear_prepare_posterior(data, model, ...
    inference, states)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

T = model.graph{1}.htheta.T;

posterior = struct('data', data, 'model', model, 'inference', inference, ...
    'samples_theta', [], 'fe', []);

np = numel(states);
nc = numel(T);

theta = cell(np, 1); 
for i = 1:np
    theta{i} = states{i}.graph{end};
end

llh = zeros(nc, np);
for i = 1:np
    llh(:, i) = sum(states{i}.llh{1}, 1);
end

fe = trapz(T, mean(llh, 2));

posterior.samples_theta = theta;
posterior.fe = fe;

end

