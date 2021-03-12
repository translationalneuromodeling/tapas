function propvalArray = tapas_uniqc_struct2propval(propvalStruct, doRemoveEmptyProps)
% Converts structure variable with property/values to cell of name/value
% pairs
%
%   propvalArray = tapas_uniqc_struct2propval(propvalStruct)
%
% IN
%   propvalStruct   struct of the form
%                   propvalStruct.prop1 = val1;
%                       ...
%                   propvalStruct.propN = valN;
%   doRemoveEmptyProps
%                   false (default) or true
%                   if true, removes all properties with empty values from
%                   output Array
% OUT
%   propvalArray    cell(1, nProps*2) of property name/value pairs, i.e., 
%                   {prop1, val1, ..., propN, valN}
%
% EXAMPLE
%   argsArray = tapas_uniqc_struct2propval(argsStruct)
%
%   See also
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-08
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%
if nargin < 2
    doRemoveEmptyProps = false;
end

propvalArray = reshape([fieldnames(propvalStruct), ...
    struct2cell(propvalStruct)].', 1, []);

if doRemoveEmptyProps
    idxEmptyProp = find(cellfun(@isempty, propvalArray(2:2:end)));
    propvalArray([2*idxEmptyProp 2*idxEmptyProp-1]) = [];
end
