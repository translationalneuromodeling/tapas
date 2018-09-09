function [posterior] = tapas_vlinear_estimate_interface(data, model, ...
    inference)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

% Verify input 
tapas_validate_data(data);
tapas_validate_model(model);
tapas_validate_inference(inference);

[data] = tapas_linear_prepare_data(data, model, inference);
[model] = tapas_vlinear_prepare_model(data, model, inference);
[inference] = tapas_vlinear_prepare_inference(data, model, inference);

[posterior] = inference.estimate_method(data, model, inference);

end

