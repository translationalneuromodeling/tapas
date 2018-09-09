function [theta] = tapas_mh_mc3_propose_gaussian_sample(data, model, ...
    inference, state, node)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

otheta = state.graph{node};
nc = numel(otheta.y);

theta = otheta;

% Default if not no update values
if isfield(state.kernel{node}{1}, 'nuk')
    nup = state.kernel{node}{1}.nuk;
else
    nup = eye(size(state.kernel{node}{1}.k, 1));
end

for i = 1:nc
    k = state.kernel{node}{i}.k;
    s = sqrt(state.kernel{node}{i}.s);
    
    theta.y{i} = otheta.y{i} + full(s .* k' * randn(size(k, 2), 1));
end

end

