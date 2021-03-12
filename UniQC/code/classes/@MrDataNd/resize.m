function outputImage = resize(this, newDimInfo)
% Resizes ND-array to a new matrix size (nSamples), given by dimInfo.
% Singleton-dimensions are replicated, while others are interpolated to the
% new number of samples in that dimension
%
%   Y = MrDataNd()
%   Y.resize(newDimInfo)
%
% This is a method of class MrDataNd.
%
% IN
%
% OUT
%
% EXAMPLE
%   resize
%
%   See also MrDataNd tapas_uniqc_resizeNd

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-06-24
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


resizedData = this.data;

nSamplesOther = size(resizedData);
nSamplesOther((end+1):newDimInfo.nDims) = 1;
iSingletonDim = find(nSamplesOther == 1); % to be replicated!

% if sizes do not match, perform 
% a) replication of singleton dimensions (i.e. 1 slice => N x replicated)
% b) interpolation of non-singleton dimensions (e.g. 5 slices => 10 slices)

factorsReplication = ones(1, newDimInfo.nDims);
factorsReplication(iSingletonDim) = newDimInfo.nSamples(iSingletonDim);

% a) replication of singleton dimensions (i.e. 1 slice => N x replicated)
resizedData = repmat(resizedData, factorsReplication);

% b) interpolation of non-singleton dimensions (e.g. 5 slices => 10 slices)
resizedData  = tapas_uniqc_resizeNd(resizedData, newDimInfo.nSamples);

% Save output, update data and dimInfo
outputImage 	 	= this.copyobj();
outputImage.data 	= resizedData;
outputImage.dimInfo = newDimInfo.copyobj;