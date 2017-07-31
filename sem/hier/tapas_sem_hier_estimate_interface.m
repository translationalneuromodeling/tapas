function [posterior] = tapas_sem_hier_estimate_interface(data, model, inference)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Verify input 
tapas_validate_data(data);
tapas_validate_model(model);
tapas_validate_inference(inference);

[data] = tapas_sem_hier_prepare_data(data, model, inference);
[model] = tapas_sem_hier_prepare_model(data, model, inference);
[inference] = tapas_sem_hier_prepare_inference(data, model, inference);

posterior = struct();

[posterior] = inference.estimate_method(data, model, inference);


end
