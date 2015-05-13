function [dn] = mpdcm_set_device(dn)
%% Sets the device.
%
% Input:
%
%   dn      -- Device number
%
% Output:
%
%   dn      -- Devece number.
%
%
% Integer representing a device.
%
% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
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
