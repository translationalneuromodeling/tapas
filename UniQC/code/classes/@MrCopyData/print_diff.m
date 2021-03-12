function s = print_diff(obj, input_obj, verbose)
% Prints differing property names along with their values
%
%   Y = MrCopyData()
%   Y.print_diff(inputs)
%
% This is a method of class MrCopyData.
%
% IN
% verbose   {true} or false; if false, only the string is created, but no output to the command window
%
% OUT
% s         cell of strings of reported non-empty values of MrCopyData-object
%
% EXAMPLE
%   print_diff
%
%   See also MrCopyData

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-04-19
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


[isObjectEqual, out_left, out_right] = obj.comp(input_obj);
if nargin  < 3
    verbose = true;
end
sl = out_left.print('',0);
sr = out_right.print('',0);

% find unique affected properties
sUniqueProps = unique([sl(:,1); sr(:,1)]);
nU = length(sUniqueProps);

s = cell(nU,3);
for c = 1:nU
    iL = find(strcmp(sUniqueProps{c}, sl(:,1)));
    iR = find(strcmp(sUniqueProps{c}, sr(:,1)));
    s{c,1} = sUniqueProps{c};
    if isempty(iL)
        s{c,2} = '[]';
    else
        s{c,2} = sl{iL,2};
    end
    if isempty(iR)
        s{c,3} = '[]';
    else
        s{c,3} = sr{iR,2};
    end
    if verbose
        fprintf('%40s: %40s   VS   %s\n', s{c,1}, s{c,2}, s{c,3});
    end
end
end