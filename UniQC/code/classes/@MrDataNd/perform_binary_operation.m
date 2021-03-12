function outputImage = perform_binary_operation(this, otherImage, ...
    functionHandle)
% Performs binary operation (i.e. 2 inputs) on this and a second image
% for given dimension and returns a new output image
%
% NOTE: If voxel dimensions of 2nd image do not match - or a scalar is
% given as 2nd argument - data in the 2nd argument is automatically
% replicated to match this image geometry.
%
%   Y = MrDataNd()
%   outputImage = perform_binary_operation(this, otherImage, ...
%   functionHandle)
%
% This is a method of class MrDataNd.
%
% IN
%   otherImage              2nd operand for binary operation
%   functionHandle          handle of function to be applied to images (e.g.
%                           @plus, @minus, @times and @rdivide for the 4 
%                           arithmetic operations )
%
% OUT
%   outputImage             new MrDataNd with possibly new image dimensions,
%                           output of binary operation performed on this
%                           and otherImage
% EXAMPLE
%
%   % Compute difference of 2 images
%		Y = MrDataNd();
%		Z = MrDataNd();
%		X = Y.perform_binary_operation(Z, @minus);
%
%	% Scale image (multiply) by a factor of 3
%		Y = MrDataNd();
%		Y = Y.perform_binary_operation(3, @mult)
%
%	% Compute ratio of 2 images
%		Y 			= MrDataNd();
%		Z 			= MrDataNd();
%		ratioYZ 	= Y.perform_binary_operation(Z, @rdivide);
%	
%
%
%   See also MrDataNd perform_unary_operation

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-13
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% binary operation with scalar etc. => make MrDataNd first
if ~isa(otherImage, 'MrDataNd')
    otherData = otherImage;
    otherName = 'dataMatrix';
    % Create image from data matrix with similar dimInfo
    % TODO: include proper dimInfo-adaptation (right now, res is assumed
    % the same, not FOV!)
    nSamplesOther = size(otherData);
    nSamplesOther(end+1:this.dimInfo.nDims) = 1; % avoid singleton dim error!
    dimInfoOther = this.dimInfo.copyobj;
    dimInfoOther.nSamples = nSamplesOther;
    
    otherImage = this.copyobj();
    otherImage.dimInfo = dimInfoOther;
    otherImage.data = otherData;
    otherImage.name = otherName;
end

%% TODO: FOV first, then adapt matrix sizes?
% TODO: Check FOVs first, if they don't match crop or zero-fill otherImage

outputImage = otherImage.resize(this.dimInfo);
outputImage.name = this.name;

%% Perform computation and store!
outputImage.data 	= functionHandle(this.data, outputImage.data);

outputImage.info{end+1,1} = sprintf('%s( %s, %s )', func2str(functionHandle), ...
    outputImage.name, otherImage.name);