function L = compute_slice_leakage(this, dy, dz)
% Computes slice leakage as cross-correlation between this image and a FOV
% and slice-shifted copy
%
%   Y = MrDataNd()
%   L = Y.compute_slice_leakage(dy, dz)
%
% This is a method of class MrDataNd.
%
% IN
%   dy  [1, nShifts], FOV shift(s) in number of pixels in y (phase encode)
%   dz  [1, nShifts], FOV shift(s) in number of pixels in z (slice direction)
%                     i.e., distance between slices that a simultaneously
%                     excited
%
% OUT
%
% EXAMPLE
%   compute_slice_leakage
%
%   See also MrDataNd
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-10-10
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

X = this;

if nargin < 3
    dz = 1:(X.dimInfo.z.samples-1);
end

if nargin < 2
    dy = 0:(X.dimInfo.y.samples-1);
end

nShiftsY = numel(dy);
nShiftsZ = numel(dz);
LArray = cell(nShiftsY, nShiftsZ);
for iShiftY = 1:nShiftsY
    for iShiftZ = 1:nShiftsZ
        % shift s.th. slice 1 is correlated with slice 1+dy
        newY = circshift(1:X.dimInfo.y.nSamples, -dy(iShiftY));
        newZ = circshift(1:X.dimInfo.z.nSamples, -dz(iShiftZ));
        Y = X.shuffle({'z','y'}, {newZ, newY});
    
        % mean-correct
        X0 = X - mean(X,'t');
        Y0 = Y - mean(Y,'t');
        
        % cross corr definition, over time!
        % sqrt correction factor, since mean is computed with 1/N, but std
        % with 1/(N-1) per default, but for identical images, we want xcorr
        % to be 1
        nVolumes = X.dimInfo.t.nSamples;
        LArray{iShiftZ,iShiftY} = mean(X0.*Y0, 't')./(std(X0, 't').*std(Y0, 't')).*(nVolumes/(nVolumes-1));
        LArray{iShiftZ,iShiftY}.dimInfo.add_dims({'dz', 'dy'}, 'samplingPoints', {iShiftZ, iShiftY}, 'units', {'voxel', 'voxel'});
    end
end
L = LArray{1,1}.combine(LArray);
