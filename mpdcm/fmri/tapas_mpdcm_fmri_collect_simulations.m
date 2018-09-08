function [y] = tapas_mpdcm_fmri_collect_simulations(container)
%% Collect the simulations from a container
%
% Input
%       container       -- A container object.
%       
% Output
%       y               -- An array with the simulations.
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

y = c_mpdcm_collect_simulations(container);
y = y';

for i = 1:numel(y)
        y{i} = y{i}';
end

end

