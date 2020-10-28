function [inference] = tapas_hgf_prepare_inference(data, model, inference)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

nc = numel(model.graph{2}.htheta.T);

inference.kernel = cell(4, 1);

inference.kernel{2} = struct('k', [eye(nb)], 's', [0.5]);
inference.kernel{3} = struct('k', [eye(nb)], 's', [0.5]);

end

