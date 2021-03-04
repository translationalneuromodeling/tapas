function [folderName, folderPath] = ...
    tapas_uniqc_find_processing_folder(MrSeriesFolder, processingStep)
% Finds the folder given a processing step in an MrSeries_folder
%
% [folder_name, folder_path] = ...
% tapas_uniqc_find_processing_folder(MrSeries_folder, processing_step)
%
% IN
%       MrSeriesFolder      path of the MrSeries
%       processingStep      processing step (name or number)
%
% OUT
%
% EXAMPLE
% tapas_uniqc_find_processing_folder('mypath\MrSeries_1', 'realign')
% tapas_uniqc_find_processing_folder('mypath\MrSeries_1', 1)
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-02-05
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% init counter
foundCount = 1;
folderName = [];
folderPath = [];
% get all folders
allFolders = dir(MrSeriesFolder);
% go through all folders and compare
for n = 1:numel(allFolders)
    % check whether match is found
    if isnumeric(processingStep)
        isFound = ~isempty(regexp(allFolders(n).name,...
            num2str(processingStep, '%03.0f'),'once'));
    else
        isFound = ~isempty(regexp(allFolders(n).name,...
            processingStep, 'once'));
    end
    
    % save folder name if found
    if isFound
        folderName{foundCount} = allFolders(n).name;
        folderPath{foundCount} = fullfile(MrSeriesFolder, ...
            folderName{foundCount});
        foundCount = foundCount + 1;
    end
end

if isempty(folderName)
    disp('Sorry. Nothing found.');
end

end