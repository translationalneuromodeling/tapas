function s = print(obj, pfx, verbose)
% Prints all non-empty values of a object along with their property names
%
%   Y = MrCopyData()
%   s = Y.print(pfx, verbose)
%
%
% This is a method of class MrCopyData.
%
% IN
% verbose   {true} or false; if false, only the string is created, but
%           no output to the command window
%
% OUT
% s         cell of strings of reported non-empty values of MrCopyData-object
%
% EXAMPLE
%   Y.print
%
%   See also MrCopyData

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-12-09
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


if nargin < 2
    pfx = '';
end
if nargin < 3
    verbose = true;
end
s = cell(0,2);

[sel, mobj] = get_properties_to_update(obj);

for k = sel(:)'
    tmps = [];
    pname = mobj.Properties{k}.Name;
    currProp = obj.(pname);
    if isa(currProp, 'MrCopyData') %recursive comparison
        tmps = currProp.print([pfx '.' pname], verbose);
    else
        % cell of MrCopyData also treated
        if iscell(currProp) ...
                && length(currProp) ...
                && isa(currProp{1}, 'MrCopyData')
            for c = 1:length(currProp)
                tmps2 = currProp{c}.print([pfx '.' pname], verbose);
                if ~isempty(tmps2), tmps = [tmps; tmps2]; end
            end
        else % no cell of MrCopyData, no MrCopyData...any other property
            if ~isempty(currProp)
                tmps{1,1} = [pfx '.' pname];
                if ischar(currProp)
                    tmps{1,2} = currProp;
                elseif iscell(currProp)
                    % print only subset of elements
                    nElementsMax = 30;
                    nElementsToPrint = min(nElementsMax,numel(currProp{1}));
                    tmps{1,2} = 'cell array ';
                    if ischar(currProp{1})
                        tmps{1,2} = [tmps{1,2} sprintf('%s ', currProp{1}(1:nElementsToPrint))];
                    else
                        tmps{1,2} = [tmps{1,2} sprintf('%f ', currProp{1}(1:nElementsToPrint))];
                    end
                elseif isstruct(currProp)
                    tmps{1,2} = 'struct';
                elseif isenum(currProp)
                    tmps{1,2} = char(currProp);
                else    
                    pp = currProp(1,1:min(size(currProp,2), 16));
                    if (floor(double(pp(1)))==ceil(double(pp(1)))) %print integers differently
                        tmps{1,2} = sprintf('%d ', pp);
                    else
                        tmps{1,2} = sprintf('%4.2e ', pp);
                    end
                    if numel(currProp)>numel(pp), tmps{1,2} = [tmps{1,2}, '...']; end
                end
                if verbose
                    fprintf('%70s = %s\n', tmps{1,1}, tmps{1,2});
                end
            end
        end
    end
    if ~isempty(tmps), s = [s; tmps]; end
end

end