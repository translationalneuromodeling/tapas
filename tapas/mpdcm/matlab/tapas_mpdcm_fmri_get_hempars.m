function [hempars] =  tapas_mpdcm_fmri_get_hempars(spmv)
%% Return constants used for the hemodynamic parameters.
%
% Input:
% spmv -- spm version. Should be a string. Only SPM8 is supported.
%
% Output:
% hempars -- Structuce containing default values for the hemodynamic parameters
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

if nargin < 1
    spmv = 'SPM8';
end

switch spmv
    case 'SPM8'
        hempars = struct('K', 0.64, 'tau', log(2.0), 'gamma', 0.32, ...
            'alpha', 0.32, 'E0', 0.32, 'V0', 4); 
    otherwise
        error('mpdcm:fmri:get_hempars:input', 'Unsupported SPM version');
end


end

