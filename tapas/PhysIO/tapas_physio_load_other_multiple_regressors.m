function R = tapas_physio_load_other_multiple_regressors(filename_other_regressors)
% reads other multiple regressors from .txt-file or as variable R from .mat
%
%   R = tapas_physio_load_other_multiple_regressors(filename_other_regressors)
%
% IN
%   filename_other_regressors   filename (.mat or .txt) of other confound
%                               regressors
%
% OUT
%   R                           [Nscans NconfoundRegressors] matrix of multiple regressors
% EXAMPLE
%   tapas_physio_load_other_multiple_regressors
%
%   See also tapas_physio_orthogonalise_physiological_regressors tapas_physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2013-02-21
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_load_other_multiple_regressors.m 354 2013-12-02 22:21:41Z kasperla $
[fp, fn, fs] = fileparts(filename_other_regressors);

if ~exist(filename_other_regressors, 'file')
    warning('No input multiple regressors found');
    R = [];
else
    rp = load(filename_other_regressors);
    if strcmp('.txt', fs) % text-file
        R = rp;
    else % mat file
        R = rp.R;
    end
end
end
