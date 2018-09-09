function [model] = tapas_linear_prepare_model(data, model, inference)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of regressors
nc = numel(model.graph{2}.htheta.T);
nb = size(data.u{1}, 2);
% Shrink priors
y = zeros(nb, 1);

model.graph{4}.htheta = y;


end

