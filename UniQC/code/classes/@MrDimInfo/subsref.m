function varargout = subsref(this, S)
% Allows subscript referencing with dot notation of dimLabels
%
%   Y = MrDimInfo()
%   Y.subsref(S)
%
% This is a method of class MrDimInfo.
% It checks whether the provided S(1).subs is
%   a) a valid dimLabel and returns the reduced MrDimInfo for this dimension (via get_dims(dimLabel)).
%       e.g. dimInfo.z.samplingPoints
%   b) a valid property of MrDimInfo and S(2).subs a valid dimLabel, which
%       is transformed into a dim-index to allow referencing
%       e.g. dimInfo.nSamples('z')
% For anything but these first-level dot notations, it uses the builtin subsref.
%
% IN
%   S   structure with two fields:
%           type is a char vector containing '()', '{}', or '.', indicating the type of indexing used.
%           NOTE: we only overload the dot notation here to index dimLabels
%           subs is a cell array or character array containing the actual subscripts.% OUT
%
% EXAMPLE
%   This enables the usage of dimLabels in the dot notation, e.g.
%
%   dimInfo = MrDimInfo('dimLabels', {'x', 'y', 'z'},
%                       'samplingPoints', {1:10, -10:-1, 5:14});
%   dimInfo.z.samplingPoints
%       => ans =
%      5     6     7     8     9    10    11    12    13    14
%
%   dimInfo.nSamples('z');
%
%   See also MrDimInfo builtin.subsref

% Author:   Lars Kasper & Saskia Bollmann
% Created:  2017-06-28
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


switch S(1).type
    case '.'
        % do custom dot-referencing  for valid dimLabels (e.g. dimInfo.x etc)
        if ismember(S(1).subs, this.dimLabels)
            
            if numel(S) == 1 % return reduced dimInfo object
                varargout = {this.get_dims(S(1).subs)};
            else
                % retrieve reduced dimInfo and continue with classical subsref
                % from there
                varargout = {builtin('subsref',this.get_dims(S(1).subs),S(2:end))};
            end
        elseif ismember(S(1).subs, properties(this)) && numel(S) > 1 && numel(S(2).subs) == 1 
                % do custom dot-referencing allowing for property(dimLabel), e.g. resolutions('x')
                % by converting char/cell indices to numerical ones and run normal
                % subsref
                % note: allows only 1D indexing, i.e. not for ranges(:, 1:2)
                S(2).subs = {this.get_dim_index(S(2).subs{:})};
                varargout = {builtin('subsref',this,S)};
        elseif ismember(S(1).subs, methods(this))
            % a method call in dot notation, e.g. dimInfo.select() shall be
            % treated correctly, even for variable number of output
            % arguments
            if nargout
                [varargout{1:nargout}] = builtin('subsref',this,S);
            else
                varargout = {builtin('subsref',this,S)};   
            end
        else % hope for the best
                varargout = {builtin('subsref',this,S)};
        end
    case '()' % allow to retrieve dimInfo('x'), dimInfo({'x','y'}), dimInfo(1:3)
        % for valid dimension labels/ indices
        sub = S(1).subs{:};
        isValidDimLabel = (ischar(sub) || iscellstr(sub)) && all(ismember(sub, this.dimLabels));
        % dimIndices have to be integer and within range of validdimensions
        isValidDimIndex = isnumeric(sub) && isequal(fix(sub),sub) ...
            && all(sub >=1) && all(sub <= this.nDims);
        if isValidDimLabel || isValidDimIndex
            varargout = {this.get_dims(S(1).subs{:})};
        else
            % use builtin indexing and hope for the best
            varargout = {builtin('subsref',this,S)};
        end
    otherwise
        % use builtin indexing for Y{i,j}
        varargout = {builtin('subsref',this,S)};
end