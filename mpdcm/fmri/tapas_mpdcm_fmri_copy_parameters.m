function [ntheta] = tapas_mpdcm_fmri_copy_parameters(ntheta, otheta, fields)
%% Transfer parameters in fields from otheta to ntheta.
%
% Input:
% ntheta     -- Cell array of parameters in structure form
% otheta     -- Hyperparameters
%
% Ouput:
% thate     -- Cell array of parameters in structure form
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%


nt = numel(ntheta);

for i = 1:nt
    for j = 1:numel(fields)
        ntheta{i} = setfield(ntheta{i}, fields{j}, ...
            getfield(otheta{i}, fields{j}));
    end
end

end


