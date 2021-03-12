function this = read_matrix_from_workspace(this, inputMatrix)
% Reads in matrix from workspace, updates dimInfo according to data
% dimensions
%
%   Y = MrDataNd()
%   Y.read_matrix_from_workspace()
%
% This is a method of class MrDataNd.
%
% IN
%
% OUT
%
% EXAMPLE
%   read_matrix_from_workspace
%
%   See also MrDataNd

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-10-12
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% check whether valid dimInfo now

% TODO: update dimInfo, but keeping information that is unaltered by
% changing data dimensions...
% e.g. via dimInfo.merge
hasDimInfo = isa(this.dimInfo, 'MrDimInfo');


this.data = inputMatrix;

% remove singleton 2nd dimension kept by size command
nSamples = size(this.data);
if numel(nSamples) == 2
    nSamples(nSamples==1) = [];
end
resolutions = ones(1, numel(nSamples));

% set dimInfo or update according to actual number of samples
if ~hasDimInfo
    this.dimInfo = MrDimInfo('nSamples', nSamples, ...
        'resolutions', resolutions);
else
    if any(nSamples) % only update dimInfo, if any samples loaded
        if (numel(nSamples) ~= this.dimInfo.nDims)
            % only display the warning of an non-empty dimInfo (i.e. nDims
            % ~=0) has been given
            if (this.dimInfo.nDims ~=0)
                warning('Number of dimensions in dimInfo (%d) does not match dimensions in data (%d), resetting dimInfo', ...
                    this.dimInfo.nDims, numel(nSamples));
            end
            this.dimInfo = MrDimInfo('nSamples', nSamples, ...
                'resolutions', resolutions);
        elseif ~isequal(this.dimInfo.nSamples, nSamples)
            % if nSamples are correct already, leave it at that, otherwise:
            
            currentResolution = this.dimInfo.resolutions;
            
            isValidResolution = ~any(isnan(currentResolution)) || ...
                ~any(isinf(currentResolution)) && ...
                numel(currentResolution) == numel(nSamples);
            
            if isValidResolution
                % use update of nSamples to keep existing offset of samplingPoints
                this.dimInfo.nSamples = nSamples;
            else % update with default resolutions = 1
                this.dimInfo.set_dims(1:this.dimInfo.nDims, 'nSamples', ...
                    nSamples, 'resolutions', resolutions);
            end
            
        end
    end
end
