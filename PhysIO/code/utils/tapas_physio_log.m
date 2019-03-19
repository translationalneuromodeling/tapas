function [verbose, msg] = tapas_physio_log(msg, verbose, warningLevel)
% writes message into verbose.process_log, prints it, and labels it as
% warning, if necessary
%
%   [verbose, msg] = tapas_physio_log(msg, verbose, isWarning)
%
% IN
%   msg         message string
%   verbose     physio.verbose structure, including verbose.process_log
%               if verbose.level < 0, output is not written to command
%               window
%   warningLevel 0 = message
%                1 = Matlab warning
%                2 = error
%
% OUT
%
% EXAMPLE
%   tapas_physio_log
%
%   See also

% Author: Lars Kasper
% Created: 2015-07-05
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 2 || isempty(verbose)
   tmpPhysio = tapas_physio_new();
   verbose = tmpPhysio.verbose;
end 

if nargin < 3
    warningLevel = 0;
end

currStack = dbstack;
verbose.process_log{end+1,1} = sprintf('%s (%s - line %d)', msg, ...
    currStack(2).file, currStack(2).line);

if verbose.level >=0 || warningLevel == 2
    switch warningLevel
        case 2
            filenameProcessLog = sprintf('tapas_physio_error_process_log_%s.mat', ...
                datestr(now, 'yyyy-mm-dd_HHMMSS'));
            try
                save(filenameProcessLog, 'verbose');
            catch
                warning('tapas:physio', ['Could not save verbose info in %s, ' ...
                    'throwing original error now:'], filenameProcessLog);
            end
            error('tapas:physio', msg);
        case 1
            warning('tapas:physio', msg);
        case 0
            fprintf('%s\n',msg);
    end
end



