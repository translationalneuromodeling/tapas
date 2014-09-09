function sample_points = tapas_physio_get_sample_points(ons_secs, sqpar, slicenum)
% gets times of slice scan events of a particular slice number in every
% volume acquired
%
% USAGE
%   sample_points = tapas_physio_get_sample_points(ons_secs, sqpar, slicenum)
%
% INPUTS:
%   ons_secs
%   sqpar
%   slicenum    - slice number (1<=slicenum<=Nslices) where signal shall be
%                 sampled
%
% OUTPUT:
%   sample_points   - absolute time (in seconds) where the specified slice was
%                     aquired for every volume
%
% Author: Lars Kasper
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_sample_points.m 235 2013-08-19 16:28:07Z kasperla $
if nargin<3 || isempty(slicenum)
    sample_points = [];
    for n = 1:sqpar.Nscans
        sample_points = [sample_points; ons_secs.spulse_per_vol{n + sqpar.Ndummies}];
    end
else
    nSampleSlices = length(slicenum);
    sample_points = zeros(sqpar.Nscans*nSampleSlices,1);
    for n = 1:sqpar.Nscans
        spulse = ons_secs.spulse_per_vol{n + sqpar.Ndummies};
        if length(spulse) < max(slicenum)
            error('scan %d: only %d slice scan events. Cannot resample to slice %d', ...
                n, length(spulse), max(slicenum));
        else
            sample_points((n-1)*nSampleSlices + (1:nSampleSlices)) = spulse(slicenum);
        end
    end
end

end
