function [ err ] = tapas_mpdcm_device_reset()
%% Resets the device.
%
% Input
%
% Output
%   err     -- Error code if any
%
% aponteeduardo@gmail.com
% copyright (C) 2015
%


err = c_mpdcm_device_reset();

end

