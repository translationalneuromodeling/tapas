function job = tapas_physio_replace_absolute_paths(job, pathArray)
% Replaces all absolute paths in a structure variable by relative ones
%
%   job = tapas_physio_replace_absolute_paths(job, pathArray)
%
% IN
%   job     structure variable, e.g. SPM, matlabbatch, or physio structure
%           with different fields (physio.save_dir,
%           physio.log_files.cardiac...)
%   pathArray cell(nPaths,1) of absolute paths in occuring in some of the
%           fields that shall be removed;
%
% NOTE: only tested on Mac/Linux systems so far, \ instead of / might lead
%       to unexpected behavior
%
% OUT
%
% EXAMPLE
%   tapas_physio_replace_absolute_paths
%
%   See also tapas_physio_fieldnamesr
%
% Author: Lars Kasper
% Created: 2015-08-10
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_replace_absolute_paths.m 797 2015-08-10 16:07:10Z kasperla $

jobFields = tapas_physio_fieldnamesr(job);

if ~iscell(pathArray)
    pathArray = {pathArray};
end

nFields = numel(jobFields);
nPaths = numel(pathArray);

for f = 1:nFields
    for p = 1:nPaths
        if ~isempty(pathArray{p})
            strField = ['job.' jobFields{f}];
            % replace strings
            if eval(['isstr(' strField ')'])
                % to lazy for not using eval :-(
                eval([ strField ' = regexprep(' strField ', ''' pathArray{p} '[/]*'', '''');']);
                % replace cellstrings
            elseif eval(['iscell(' strField ')']) && eval(['isstr(' strField '{1})'])
                eval([ strField '{1} = regexprep(' strField '{1}, ''' pathArray{p} '[/]*'', '''');']);
            end
        end
    end
end