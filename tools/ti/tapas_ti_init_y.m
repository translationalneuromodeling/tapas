function [ny] = tapas_ti_init_y(y, ptheta, pars)
%% Default initilization.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

ny = cell(1, numel(pars.T));
ny(:) = {y};

end

