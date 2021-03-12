function isObjectEqual = tapas_uniqc_isequaltol(a,b, tolerance)
% Performs isequal similar to matlab for numeric values, strings and cells,
% but allows for a tolerance (also for cell of cells, recursive compare)
%
%  isObjectEqual = tapas_uniqc_isequaltol(a,b, tolerance)
%
% NOTE: the odd cases empty, nan and inf are all considered equal, for
% objects, their nativ isequal method is tried with and then without a
% tolerance argument
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_isequaltol
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-02-27
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

if nargin < 3
    tolerance = eps('single');
end

%% First, deal with pure numeric and string comparisons, then the recursive
% cases (struct, cell)
if isnumeric(a) && isnumeric(b)
    
    % the odd cases: empty, nan and infs in comparison
    hasSpecialValues = isempty(a) || isempty(b) || ...
        any(isnan(a)) || any(isnan(b)) || ...
        any(isinf(a)) || any(isinf(b));
    
    if hasSpecialValues
        % the odd cases empty, nan and inf are all considered equal
        isObjectEqual = (isempty(a) && isempty(b)) || ...
            ~any(isnan(a(:)) - isnan(b(:))) || ... % no differences in NaN status per element
            ~any(isinf(a(:)) - isinf(b(:))); % no differences in NaN status
    else
        % numeric comparisons, no class distinction necessary
        % int32, double, single etc. are all considered equal, if values are
        % the same
        isObjectEqual = isequal(size(a), size(b)) && ...
            (all(abs(a(:)-b(:)) <= tolerance));
    end
else
    
    classA = class(a);
    classB = class(b);
    
    % class distinction helpful, but strings and chars are treated equal by
    % Matlab's isequal
    isString = ismember(classA, {'string', 'char'}) && ...
        ismember(classB, {'string', 'char'});
    
    if isString
        isObjectEqual = isequal(a, b); % no tolerance for strings!
    else
        % we need the same class for all other cases to be equal, and the
        % same size to enable the recursion
        isObjectEqual = isequal(classA, classB) && ...
            isequal(size(a), size(b));
        
        if isObjectEqual
            
            switch classA
                case 'cell'
                    n = 0;
                    nElements = numel(a);
                    while isObjectEqual && n < nElements
                        n = n + 1;
                        isObjectEqual = isObjectEqual && ...
                            tapas_uniqc_isequaltol(a{n}, b{n}, tolerance);
                    end
                case 'struct'
                    fieldNames = fields(a);
                    nFields = numel(fieldNames);
                    n = 0;
                    while isObjectEqual && n < nFields
                        n = n + 1;
                        isObjectEqual = isObjectEqual && ...
                            tapas_uniqc_isequaltol(a.(fieldNames{n}), ...
                            b.(fieldNames{n}), tolerance);
                    end
                otherwise
                    % Maybe we are lucky and this is an object with an
                    % implemented isequal method
                    try
                        isObjectEqual = isequal(a,b,tolerance);
                    catch
                        try
                            isObjectEqual = isequal(a,b);
                        catch
                            isObjectEqual = 0; % save option, at least gives errors
                        end
                    end
            end
        end
    end
end