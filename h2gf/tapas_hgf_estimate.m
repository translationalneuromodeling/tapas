function [posterior] = tapas_hgf_estimate(hgf, model, inference, pars)
%% Estimates the hgf using mcmc.
%
% Input
%   hgf         -- Configured hgf.
%   model       -- Model object.
%   inference   -- Inference object.
%   pars        -- Parameter structure.
%       
% Output
%  posterior    -- Structure containing the posterior.
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

[pars] = tapas_hgf_pars(pars);
[data] = tapas_hgf_data(hgf, pars);
[model] = tapas_hgf_model(hgf, pars);
[inference] = tapas_hgf_inference(inference, pars);

[posterior] = tapas_hgf_estimate_interface(data, model, inference);


end

