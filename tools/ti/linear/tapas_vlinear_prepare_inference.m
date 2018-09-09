function [inference] = tapas_vlinear_prepare_inference(data, model, inference)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

inference = tapas_linear_prepare_inference(data, model, inference);

inference.kernel{3} = [];

end

