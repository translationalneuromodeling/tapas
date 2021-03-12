function [this, hasRegressors, hasConditions]  = init_glm(this)
% This method initializes MrGlm, i.e. checks for consistency and saves
% neccessary files
%
%   Y = MrGlm()
%   Y.specify_Glm(inputs)
%
% This is a method of class MrGlm.
%
% IN
%
% OUT
%
% EXAMPLE
%   init_glm
%
%   See also MrGlm

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-07
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


% save regressor file
R = struct2array(this.regressors);
if ~isempty(R)
    R = struct2array(this.regressors);
    fileNameRegressors = fullfile(this.parameters.save.path, 'Regressors');
    save(fileNameRegressors, 'R');
    hasRegressors = 1;
else
    disp('No regressors specified. Are you sure?');
    hasRegressors = 0;
end

% save conditions file
if ~isempty(this.conditions.names)
    names = this.conditions.names;
    onsets = this.conditions.onsets;
    durations = this.conditions.durations;
    fileNameConditions = fullfile(this.parameters.save.path, 'Conditions');
    save(fileNameConditions, 'names', 'onsets', 'durations');
    hasConditions = 1;
else
    disp('No conditions specified. Are you sure?');
    hasConditions = 0;

end

% make SPM directory
spmDirectory = fullfile(this.parameters.save.path, this.parameters.save.spmDirectory);
if ~exist(spmDirectory)
    mkdir(spmDirectory);
end
