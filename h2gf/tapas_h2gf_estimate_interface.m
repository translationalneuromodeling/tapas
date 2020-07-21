function [posterior] = tapas_h2gf_estimate_interface(data, model, inference)
%%  
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

tapas_h2gf_validate_data(data, model, inference);
tapas_validate_model(model);
tapas_validate_inference(inference);

[data] = tapas_h2gf_prepare_data(data, model, inference);
[model] = tapas_h2gf_prepare_model(data, model, inference);
[inference] = tapas_h2gf_prepare_inference(data, model, inference);

[posterior] = inference.estimate_method(data, model, inference);


end

