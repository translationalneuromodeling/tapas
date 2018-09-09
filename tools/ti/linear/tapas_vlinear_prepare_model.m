function [model] = tapas_vlinear_prepare_model(data, model, inference)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Number of regressors
nc = numel(model.graph{2}.htheta.T);
nb = size(data.u{1}, 2);

if ~isstruct(model.graph{4}.htheta.y)
    model.graph{4}.htheta.y = struct();
end

if ~isfield(model.graph{4}.htheta.y, 'mu')
    model.graph{4}.htheta.y.mu = zeros(nb, 1);
end
if ~isfield(model.graph{4}.htheta.y, 'k')
    model.graph{4}.htheta.y.k = 1.5; % Wide prior
end
if ~isfield(model.graph{4}.htheta.y, 't')
    model.graph{4}.htheta.y.t = 0.1;
end


end

