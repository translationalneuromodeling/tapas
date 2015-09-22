function [nd] =  tapas_mpdcm_num_devices()
%% Get the number of devices.
%
% Input:
%
% Output:
% nd        -- Scalar. Total number of devices.
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

nd = c_mpdcm_num_devices();

end
