function this = subsasgn(this, S, newValue)
% Allows subscript assignment with dot notation of dimLabels
%
%
%   Y = MrDimInfo()
%   Y.subsref(S)
%
% This is a method of class MrDimInfo.
% It checks whether the provided S(1).subs is 
%   a) a valid dimLabel and updates the reduced MrDimInfo for this dimension (via get_dims(dimLabel)). 
%       e.g. dimInfo.z.samplingPoints = {5:10}
%   b) a valid property of MrDimInfo and S(2).subs a valid dimLabel, which
%       is transformed into a dim-index to allow assignment
%       e.g. dimInfo.nSamples('z') = 100
% For anything but these first-level dot notations, it uses the builtin subsref.
%
% IN
%   S   structure with two fields:
%           type is a char vector containing '()', '{}', or '.', indicating the type of indexing used.
%           NOTE: we only overload the dot notation here to index dimLabels
%           subs is a cell array or character array containing the actual subscripts.
%   newValue
%           Input arguments to be set, e.g.
% OUT
%   this    updated object
%
% EXAMPLE
%   dimInfo.z.nSamples = 20;
%   dimInfo.nSamples('y') = 100;
%   dimInfo.z.samplingPoints = {1:30};
%   dimInfo.samplingPoints({'z', 'x'}) = {1:20, 4:50}
%
%
%   See also MrDimInfo builtin.subsasgn

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-06-29
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
        % do custom dot-referencing  for valid dimLabels
        if ismember(S(1).subs, this.dimLabels)
            
            if numel(S) == 2 % S(1).subs is the dimLabel, S(2).subs
                this = this.set_dims(S(1).subs, S(2).subs, newValue);
            else
                % retrieve reduced dimInfo and continue with classical subsref
                % from there
                this = builtin('subsasgn',this.get_dims(S(1).subs),S(2:end), newValue);
            end
        else
            if ismember(S(1).subs, properties(this)) && numel(S) > 1
                % do custom dot-referencing allowing for property(dimLabel), e.g. resolutions('x')
                % by converting char/cell indices to numerical ones and run normal
                % subsref
                S(2).subs = {this.get_dim_index(S(2).subs{:})};
            end
            this = builtin('subsasgn',this,S, newValue);
        end
    otherwise
        % use builting indexing for Y(i,j) or Y{i,j}
        this = {builtin('subsasgn',this, S, newValue)};
        
end