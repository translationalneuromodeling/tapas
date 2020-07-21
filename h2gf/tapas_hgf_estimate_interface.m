function [posterior] = tapas_hgf_estimate_interface(data, model, inference)
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

[data] = tapas_hgf_prepare_data(data, model, inference);
[model] = tapas_hfg_prepare_model(data, model, inference);
[inference] = tapas_hfg_prepare_inference(data, model, inference);                                                                                          

[posterior] = inference.estimate_method(data, model, inference);


end

