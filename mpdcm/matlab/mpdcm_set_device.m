function [dn] = mpdcm_set_device(dn)
%% Sets the device.
%
% Integer representing a device.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
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
