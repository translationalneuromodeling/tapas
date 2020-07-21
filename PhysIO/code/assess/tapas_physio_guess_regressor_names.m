function [column_names] = tapas_physio_guess_regressor_names(model, R)
%tapas_physio_guess_regressor_names | Reconstructs missing names
%   Given a physio.model structure returns a best guess for the names of
%   the regressors. Useful for backwards compatibility for versions less
%   than v7.3.0.
%
% INPUT:
%   model        - physio.model structure
%   model        - physio.model.R array
%
% OUTPUT:
%   column_names - cell array of names for regressors in model.R

% Author: Sam Harrison
% Created: 2020-07-10
% Copyright (C) 2020 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% See tapas_physio_main_create_regressors.m
% Assumes physio.model.R was generated using the same order as there
column_names = {};
n_total = size(R, 2);

if model.retroicor.include
    column_names = [column_names, ...
        repmat({'RETROICOR (cardiac)'}, 1, 2 * model.retroicor.order.c), ...
        repmat({'RETROICOR (respiratory)'}, 1, 2 * model.retroicor.order.r), ...
        repmat({'RETROICOR (multiplicative)'}, 1, 4 * model.retroicor.order.cr), ...
        ];
end

if model.hrv.include
    column_names = [column_names, ...
        repmat({'HR * CRF'}, 1, numel(model.hrv.delays)), ...
        ];
end

if model.rvt.include
    column_names = [column_names, ...
        repmat({'RVT * RRF'}, 1, numel(model.rvt.delays)), ...
        ];
end

if model.noise_rois.include
    column_names = [column_names, ...
        repmat({'Noise ROIs'}, 1, sum(model.noise_rois.n_components)), ...
        ];
end

% Can have a variable number here, so bail out. This is because there is
% no way of reconstructing how many regressors were in the extra file
if model.other.include
    n_remaining = n_total - numel(column_names);
    column_names = [column_names, ...
        repmat({'Unknown'}, 1, n_remaining), ...
        ];
    return
end

if model.movement.include
    column_names = [column_names, ...
        repmat({'Movement'}, 1, model.movement.order), ...
        ];
    
    % Unkown number of outliers, but they are all that's left
    if ~strcmpi(model.movement.censoring_method, 'none')
        n_remaining = n_total - numel(column_names);
        column_names = [column_names, ...
            repmat({'Motion outliers'}, 1, n_remaining), ...
            ];
    end
end

assert( ...
    n_total == numel(column_names), ...
    'Unable to reconstruct the correct number of column names!')

end