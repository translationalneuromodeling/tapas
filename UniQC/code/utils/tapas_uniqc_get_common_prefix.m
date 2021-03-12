function pfx = tapas_uniqc_get_common_prefix(fileArray)
% determines tapas_uniqc_common prefix in list of strings
%
%   pfx = tapas_uniqc_get_common_prefix(fileArray)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_get_common_prefix
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-11-08
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


nChars = numel(fileArray{1});
isCommon = true;
iPfx = 0;

while isCommon
    
    iPfx = iPfx + 1;
    
    isCommon = all(cell2mat(cellfun(@(x) strcmp(x(1:iPfx), fileArray{1}(1:iPfx)), fileArray,'UniformOutput', false)));
    
end
iPfx = iPfx - 1;

pfx = fileArray{1}(1:iPfx);