function combinedObject = combine(this, combineHandle, tolerance)
% Combines properties or outputs of methods applied to objectArray data
% into a new object
%
%   Y = MrCompare()
%   combinedObject = combine(this, combineHandle)
%
% This is a method of class MrCompare.
%
% IN
%   combineHandle   property of the objects in data field
%                   e.g. MrSeries.snr or MrImage.rois{1}
%                       OR
%                   method handle valid for objects in data field
%                   e.g. 'max or 'kfilter' for MrImage
%
% OUT
%   combinedObject  new object combining either properties from all objects
%                   in data Array or outputs of method handles of
%                   The dimInfo of combinedObject will have the same extra
%                   dimensions as the compare object, e.g., subjects
% EXAMPLE
%   %% Combine tsnr properties from all MrSeries runs of a subject
%   extraDimInfo = MrDimInfo('dimLabels', 'session', 'samplingPoints', {[1 2 5]});
%   C = MrCompare(seriesArray, extraDimInfo);
%   tSnrCombined = C.combine('snr');
%   tSnrCombined.dimInfo.session % should display which sessions in tSnrImage
%   tSnrCombined.plot('session', 2);
%
%   %% Combine maximum intensity projection from tSNR images of all runs
%   % and subjects
%   extraDimInfo = MrDimInfo('dimLabels', {'run', 'subject'}, 'samplingPoints', {[1 2], [55 110 23});
%   C = MrCompare(tsnrImageArray, extraDimInfo);
%   maxIpCombined = C.combine('maxip');
%   maxIpCombined.dimInfo.subject % should display which subjects in tSnrImage
%   % plots maxIp of run 2 for all subjects
%   % (as montage/tile plot over subjects)
%   maxIpCombined.plot('run',2);

%   See also MrCompare

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-02-25
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 3
    tolerance = eps('single');
end


if this.dimInfo.nDims == 1
    objectArray = cell(this.dimInfo.nSamples,1);
else
    objectArray = cell(this.dimInfo.nSamples);
end
nElements = prod(this.dimInfo.nSamples);
for iObject = 1:nElements
    if isprop(this, combineHandle)
        objectArray{iObject} = this.data{iObject}.(combineHandle);
    else
        if ischar(combineHandle)
            funHandle = str2func(combineHandle);
        else
            funHandle = combineHandle;
        end
        objectArray{iObject} = funHandle(this.data{iObject});
    end
end

%% now convert into a single object, if possible

combinedObject = objectArray;

switch class(objectArray{1})
    case {'MrImage', 'MrImageSpm4D', 'MrDataNd'}
        combinedObject = this.data{1}.combine(this.data, ...
            this.dimInfo.dimLabels, tolerance);
    case 'MrRoi'
    case 'MrSeries'
    otherwise
        % nothing happens, keep an array as is
end