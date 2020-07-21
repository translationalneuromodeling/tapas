function [unwrapped] = tapas_h2gf_unwrapp_parameters(values, hgf)
%% Unwrap parameters into the more natural structure. 
%
% Input
%       values      --  Matrix of N parameters and M samples
%       hgf         --  hgf structure constructed by the model.
%
% Output
%       unwrapped      -- Unwrapped parameters.
%

% aponteeduardo@gmail.com
% copyright (C) 2018
%

unwrapped = hgf.p0 + hgf.jm * values;

end
