function [posterior] = tapas_linear_estimate(y, x, model, inference, pars)
%% Estimate a linear model 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

n = 3;
if nargin < n
    model = struct();
end

n = n + 1;
if nargin < n
    inference = struct();
end

n = n + 1;
if nargin < n
    pars = struct();
end

[pars] = tapas_linear_pars(pars);
[data] = tapas_linear_data(y, x, pars);
[model] = tapas_linear_model(model, pars);
[inference] = tapas_linear_inference(inference, pars);

[posterior] = tapas_linear_estimate_interface(data, model, inference);

end

