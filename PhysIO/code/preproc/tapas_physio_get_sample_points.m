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
%                 sampled; alternative: specify sqpar.onset_slice
%
% OUTPUT:
%   sample_points   - absolute time (in seconds) where the specified slice was
%                     aquired for every volume

% Author: Lars Kasper
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% slicenum should be field of onset_slice
if nargin<3
    if isfield(sqpar, 'onset_slice')
        slicenum = sqpar.onset_slice;
    end
end

% default timing: first slice
if isempty(slicenum)
    slicenum = 1;
end

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
