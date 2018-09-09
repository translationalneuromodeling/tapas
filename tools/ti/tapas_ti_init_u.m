function [nu] = tapas_ti_init_u(u, ptheta, pars)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nu = cell(1, numel(pars.T));
nu(:) = {u};

end

