function this = undo(this, doDeleteLastStep)
% restores previous processing step status of MrSeries, with optional deletion
%   Y = MrSeries()
%   Y.undo(doDeleteLastStep)
%
% This is a method of class MrSeries.
%
% IN
%   doDeleteLastStep    false - last step is kept for possible restore
%                       true  - last step is removed from history,
%                               including files
% OUT
%
% EXAMPLE
%   Y = MrSeries();
%   Y.realign();
%   Y.coregister();
%   Y.undo();  % restore Y-status before co-registration, keep coreg for restore
%   Y.redo();  % restore Y-status after co-registration
%   Y.undo(1); % restore Y-status before co-registration, delete
%                co-registration
%
%   See also MrSeries MrSeries.restore();

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-12-07
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


this.restore(this.nProcessingSteps-1, 1, doDeleteLastStep);