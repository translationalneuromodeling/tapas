function [dn] = tapas_mpdcm_get_device()
%% Return an integer referring to the new device.
%
% Input:
%
% Output:
% dn        -- Scalar. Device number.
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

dn = c_mpdcm_get_device();


end
