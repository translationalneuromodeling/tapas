function this = t_filter(this, cutoffSeconds)
% high-pass filters temporally (4th dimension of the image) as SPM
%
%   MrImage = t_filter(MrImage)
%
% This is a method of class MrImage.
%
% IN
%   cutoffSeconds   slower drifts than this will be filtered out
%
% OUT
%
% EXAMPLE
%   t_filter
%
%   See also MrImage spm_filter

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-02
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% convert to 2D
nVoxel = this.geometry.nVoxels;
Y = reshape(this.data, [], nVoxel(4))'; % Y = [nVolumes, nVoxel]
nVoxel3D = prod(nVoxel(1:3));

% create K for spm_filter and do it
K.RT = this.geometry.TR_s;
K.HParam = cutoffSeconds;
K.row = 1:nVoxel(4);

% spm_filter assumes Y = [nVolumes, nVoxel] dimensions
% K.row is specified to enable different filtering for different time
% frames e.g. sessions, to not filter drifts between session time gaps
Y = spm_filter(K, Y);

% back-conversion to 4D image
this.data = reshape(Y', nVoxel);
