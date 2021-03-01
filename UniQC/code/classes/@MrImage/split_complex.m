function otherImage = split_complex(this, partitionPair)
% Splits complex data in magnitude/phase or real/imag part as extra dim
% 'complex_mp' or 'complex_ri'
%
%   Y = MrImage()
%   Y.split_complex(partitionPair)
%
% This is a method of class MrImage.
%
% IN
%   partitionPair   'mp'    for magnitude/phase (default)
%                   'ri'    for real/imaginary part
%   
% OUT
%   otherImage      real-valued n+1 dimensional image, with last dimension
%                   'complex_mp' or 'complex_ri', depending on
%                   partitionPair
%   
% EXAMPLE
%   split_complex
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-22
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 2
    partitionPair = 'mp';
end

switch lower(partitionPair)
    case {'mp', 'cpx_mp', 'complex_mp'}
        dimLabel = 'complex_mp';
        partFunctionHandle{1} = @abs;
        partFunctionHandle{2} = @angle;
    case {'ri', 'cpx_ri', 'complex_ri'}
        dimLabel = 'complex_ri';
        partFunctionHandle{1} = @real;
        partFunctionHandle{2} = @imag;
end

% create magn/phase or real/imag parts of image as separate images and
% combine afterwards
otherImagePart = cell(2,1);
for iPart = 1:2
    otherImagePart{iPart} = partFunctionHandle{iPart}(this.copyobj);
    otherImagePart{iPart}.dimInfo.add_dims(dimLabel, 'units', 'nil', 'samplingPoints', {iPart});
end

otherImage = otherImagePart{1}.combine(otherImagePart, dimLabel);