function [compflag] = tapas_mpdcm_compflag()
%% Returns the precision in which mpdcm has been compiled.
%
% Input
%
% Output
%   compflag        -- 0 if mpdcm has been compiled in single precision and 1
%                   if it has been compiled in double precision.
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

compflag = c_mpdcm_compflag();

end
