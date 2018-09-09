function [data] = tapas_linear_data(y, u, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%


tapas_linear_validate_data(y, u);

data = struct('y', [], 'u', []);
data.y = y;
data.u = u;

end

