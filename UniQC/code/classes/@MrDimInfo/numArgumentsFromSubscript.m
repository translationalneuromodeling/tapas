% returns number of arguments (nargout) for method-call to dimInfo to avoid
% subsref-error
%
%   Y = MrDimInfo()
%   function n = numArgumentsFromSubscript(Y, struct, indexingContext)
%
% This is a method of class MrDimInfo.
%
% Details on issue and solution:
%   https://stackoverflow.com/questions/36714192/how-to-overload-subsref-numargumentsfromsubscript-for-functions-which-have-zer
%
% IN
%
% OUT
%
% EXAMPLE
%   numArgumentsFromSubscript
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-10-05
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

%// Check if we are calling obj.method
function n = numArgumentsFromSubscript(obj, struct, indexingContext)
if strcmp(struct(1).type, '.') && ...
        ischar(struct(1).subs) && ismember(struct(1).subs, methods(obj))
    
    %// Determine the package (if any)
    cls = class(obj);
    parts = regexp(cls, '\.', 'split');
    
    %// Import the package (if any) just into the namespace of this method
    if numel(parts) > 1
        import(cls);
    end
    
    %// Determine number of outputs for this method
    isCalledWithoutBrackets = numel(struct) == 1; % brackets would be struct(2) with type '()'
    if isCalledWithoutBrackets
        n = numel(obj);
    else
        n = nargout(sprintf('%s>%s.%s', parts{[end end]}, struct(1).subs));
    end
else
    %// Default to numel(obj)
    n = numel(obj);
end
end