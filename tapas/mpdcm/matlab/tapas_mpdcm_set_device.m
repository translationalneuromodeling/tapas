function [dn] = tapas_mpdcm_set_device(dn)
%% Sets the device.
%
% Input:
% dn      -- Device number
%
% Output:
% dn      -- Devece number.
%
%
% Integer representing a device.
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


assert(isnumeric(dn), 'mpdcm:set_device:input:not_numeric', ...
    'Input should be numeric');
assert(isa(dn, 'double'), 'mpdcm:set_device:input:not_double', ...
    'Input should be double');
assert(isscalar(dn), 'mpdcm:set_device:input:not_scalar', ...
    'Input should be scalar');
assert(isreal(dn), 'mpdcm:set_device:input:not_scalar', ...
    'Input should be real');

c_mpdcm_set_device(dn);

end
