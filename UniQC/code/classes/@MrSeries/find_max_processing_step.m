function [iProcessingStep, dirProcessing] = ...
    find_max_processing_step(this)
% Returns maximum processing step that has been done for this object (via
% folder structure, even if current object is at an earlier processing
% step)
%
%   Y = MrSeries()
%   Y.find_max_processing_step(inputs)
%
% This is a method of class MrSeries.
%
% IN
%
% OUT
%
% EXAMPLE
%   find_max_processing_step
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-28
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% find all processing directories via their 00x_ naming pattern
% find the last one and return its number as maximum processing step
foundFiles = dir(this.parameters.save.path);
fileArray = {foundFiles.name}'; % assembles all file names
isDirArray =  {foundFiles.isdir}';
dirArray = fileArray(cell2mat(isDirArray)); % selects all sub-directories in MrSeries-path

% finds all indices in dirArray where directory name starts with 3
% digits and _, i.e. the processing directory convention
indProcessingDirs = ~cellfun(@isempty, (regexp(dirArray, '^\d\d\d_')));
dirProcessingArray = dirArray(indProcessingDirs);

indDirsWithMrObject = cell2mat(cellfun(@(x) exist(fullfile(...
    this.parameters.save.path, x, 'MrObject.mat'), ...
    'file'), dirProcessingArray, 'UniformOutput', false));

dirProcessingArrayCompleted = dirProcessingArray(find(indDirsWithMrObject));

% convert directory name's first 3 chars into the processing step
% number
iProcessingStep = str2num(dirProcessingArrayCompleted{end}(1:3));


%iProcessingStep = this.nProcessingSteps;

% check for all directories with naming convention
dirProcessing =  dirProcessingArray{end};
