function this = load(this, fileName, varargin)
% loads object from struct in .mat-file, different variable names possible
%
%
%   Y = MrCopyData()
%   Y.load(filName)
%   Y.load(filName, variableName)
%
%   Options for variable name
%   1) variableName is given in varargin - load if available
%   2a) no variableName is given, and only one variable stored in fileName,
%       load this one regardless of name
%   2b) no variableName is given, and several variables stored in fileName,
%       check whether 'objectAsStruct' XOR shortened  class name (without
%       'Mr', e.g. dimInfo) is available, load this
%   2c) no variableName is given, and several variables stored in fileName,
%       'objectAsStruct' and shortened class name available, nothing is
%       loaded
%   2d) no variableName is given, and several variables stored in fileName,
%       neither 'objectAsStruct' nor shortened class name available,
%       nothing is loaded
%
% This is a method of class MrCopyData.
%
% IN
%
% OUT
%
% EXAMPLE
%   load
%
%   See also MrCopyData MrCopyData.save MrCopyData.update_properties_from

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-08-14
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% get content of file
matObj = matfile(fileName);
details = whos(matObj);
nVars = numel(details);

% retrieve variable name if given
if nargin > 2
    objectName = varargin{1};
    hasVariableName = 1;
else
    hasVariableName = 0;
end


% search for possible matches if no variable name is given
if ~hasVariableName
    
    % if only one variable stored, use that one
    if nVars == 1
        objectName = details(1).name;
    else
        % check whether any of the following is available
        hasObjectAsStruct = strcmpi('objectAsStruct', {details(:).name});
        classNameThis = class(this);
        hasClassName = strcmpi(classNameThis(3:end), {details(:).name});
        if any(hasObjectAsStruct) && ~any(hasClassName)
            objectName = 'objectAsStruct';
            % has object has struct but not class Name
        elseif ~any(hasObjectAsStruct) && any(hasClassName)
            objectName = details(hasClassName).name;
            % has class name but not object as strcut
        else % has either both or none
            warning('uniqc:MrCopyData:load', ...
                ['Nothing has been loaded. Matching variables could not be found. \n', ...
                'Either no matching variable name or too many. ', ...
                'Valid variable names are ''objectAsStruct'' and the name of the class. \n', ...
                'You can also use an explicit variable name: dimInfo.load(fileName, varName)']);
            return;
        end
    end
end
% read data
objectAsStruct = matObj.(objectName);
% update properties
this.update_properties_from(objectAsStruct);