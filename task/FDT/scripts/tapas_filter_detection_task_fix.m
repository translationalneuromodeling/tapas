%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% FIX FOR BREATHING FILTER DETECTION TASK %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% -------------------------------------------------------------------------
% Author: Olivia Harrison
% Created: 14/08/2018
%
% This software is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the 
% Free Software Foundation, either version 3 of the License, or (at your 
% option) any later version. This software is distributed in the hope that 
% it will be useful, but WITHOUT ANY WARRANTY; without even the implied 
% warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
% GNU General Public License for more details: http://www.gnu.org/licenses/
% -------------------------------------------------------------------------

% This function is provided to fix any input mistakes made whilst
% performing the FDT, or for combining two files from the same participant 
% into one file for analysis.

% To call the function, type tapas_filter_detection_task_fix into the 
% matlab terminal from the main FDT folder.

% The use of this scripts is for the following scenarios:
%   1) If any trials were incorrectly entered by the experimenter during
%      the task, use the option 'fix' when prompted by this function. This
%      will allow you to over-write specified trial results for either one
%      or both of the perception response given or the confidence score.
%      Call the function separately for each trial you wish to correct.
%   3) If the task is exited for any reason, it can be restarted (without
%      practice and calibration trials), with filter number and remaining
%      trials specified by the experimenter. Once the task is complete, the
%      two sets of results can be combined using the option 'combine' when
%      prompted by this function. In this cade, the original data files 
%      will be moved and saved into a directory called 'original_files'
%      within the results directory, and a new (combined) file will be
%      created.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALL THE FUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tapas_filter_detection_task_fix()

% Display function on screen
fprintf('\n______________________________________\n\n      FIX FILTER TASK RESULTS\n______________________________________\n\n');

% Check folder location is main FDT folder
[~,dir_name] = fileparts(pwd);
if ~strcmp(dir_name,'FDT')
   error('Not currently in main FDT folder. Please move to FDT folder and try again.');
end

% Ask for option to fix results or combine trial sets
option = input('Option (fix or combine) = ','s'); % Get option


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FIX RESPONSES OR CONFIDENCE SCORES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try
if strcmp(option,'fix') == 1
    % Load results to alter
    PPID = input('PPID = ','s'); % Get subject ID
    resultsFile = fullfile('results', ['filter_task_results_', PPID, '.mat']); % Create results file name
    load(resultsFile); % Load results
    % Choose trials to override
    a = 0;
    while a < 1
        changeTrialNum = input('Trial number to change = '); % Get trial number
        if results.setup.taskType == 1 && results.thresholdTrials.filters(changeTrialNum) == 1
            fprintf('______________________________________\n\nFilters were present for trial %d\n______________________________________\n\n', changeTrialNum);
        elseif results.setup.taskType == 1 && results.thresholdTrials.filters(changeTrialNum) == 0
            fprintf('______________________________________\n\nFilters were absent for trial %d\n______________________________________\n\n', changeTrialNum);
        elseif results.setup.taskType == 2
            fprintf('______________________________________\n\nFilters were presented in interval %d for trial %d\n______________________________________\n\n', results.thresholdTrials.interval(changeTrialNum), changeTrialNum);
        end
        changeResponse = input('Response = '); % Change response
        changeConfidence = input('Confidence = '); % Change confidence score
        % Ask for confirmation
        confirm = input('Confirm (Yes = 1, No = 0) = '); % Get confirmation
            if confirm == 1
                results.thresholdTrials.response(changeTrialNum) = changeResponse;
                results.thresholdTrials.confidence(changeTrialNum) = changeConfidence;
                results.thresholdTrials.adjustmentDateAndTime = datestr(now, 'yyyy_mm_dd_HHMMSS'); % Mark the date the results were manually adjusted
                % Re-score replaced trials
                if (results.setup.taskType == 1 && results.thresholdTrials.filters(changeTrialNum) == results.thresholdTrials.response(changeTrialNum)) || (results.setup.taskType == 2 && results.thresholdTrials.interval(changeTrialNum) == results.thresholdTrials.response(changeTrialNum))
                    results.thresholdTrials.score(changeTrialNum) = 1;
                elseif (results.setup.taskType == 1 && results.thresholdTrials.filters(changeTrialNum) ~= results.thresholdTrials.response(changeTrialNum)) || (results.setup.taskType == 2 && results.thresholdTrials.interval(changeTrialNum) ~= results.thresholdTrials.response(changeTrialNum))
                    results.thresholdTrials.score(changeTrialNum) = 0;
                end
                results.thresholdTrials.accuracyTotal = round(mean(results.thresholdTrials.score) * 100); % Re-calculate total accuracy
                fprintf('______________________________________\n\nTrial %d adjusted\n______________________________________\n\n', changeTrialNum);
            else
                fprintf('______________________________________\n\nTrial %d NOT adjusted\n______________________________________\n\n', changeTrialNum);
            end
        % Ask for further trials
        try
            more = input('Change another trial (Yes = 1, No = 0) = '); % Get confirmation
        catch
        end
        if more == 0
            outputDir = exist(fullfile('results', 'original_files')); % Check if folder for old files already exists
            if outputDir == 0
                mkdir results original_files % Make the folder if it does not exist
            elseif outputDir == 7
            end
            resultsFile_orig = fullfile('results', 'original_files', ['filter_task_results_', PPID, '_orig.mat']);
            movefile(resultsFile, resultsFile_orig); % Save a copy of the original results
            save(resultsFile, 'results'); % Save results
            fprintf('______________________________________\n\nFINISHED!\n______________________________________\n\n');
            a = 1;
        end
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMBINE THRESHOLD TRIALS IN TWO FILES TO CREATE NEW OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif strcmp(option,'combine') == 1
    % Load results to alter
    file1 = input('File 1 PPID = ','s'); % Get subject ID for first file
    resultsFile1 = load(fullfile('results', ['filter_task_results_', file1, '.mat'])); % Create and load results file 1
    file2 = input('File 2 PPID = ','s'); % Get subject ID for second file
    resultsFile2 = load(fullfile('results', ['filter_task_results_', file2, '.mat'])); % Create and load results file 2
    output = input('Output PPID = ','s'); % Get output ID for final results
    outputFile = fullfile('results', ['filter_task_results_', output, '.mat']); % Create output file and location
    fprintf('\nConcatenating File 1 + File 2 into Output file \n');
    % Make compatible with previous versions with set staircase
    if ~isfield('staircaseType', resultsFile1.results.setup)
        resultsFile1.results.setup.staircaseType = 1;
    end
    if ~isfield('staircaseType', resultsFile2.results.setup)
        resultsFile1.results.setup.staircaseType = 1;
    end
    % Concatenate all variables
    if (resultsFile1.results.setup.staircaseType == 1 && resultsFile2.results.setup.staircaseType == 1 && resultsFile1.results.thresholdTrials.filterNum == resultsFile2.results.thresholdTrials.filterNum) || (resultsFile1.results.setup.staircaseType == 2 && resultsFile2.results.setup.staircaseType == 2)
        if resultsFile1.results.setup.staircaseType == 1 && resultsFile2.results.setup.staircaseType == 1
            results.setup.staircaseType = 1;
            results.thresholdTrials.filterNum = resultsFile1.results.thresholdTrials.filterNum;
        elseif resultsFile1.results.setup.staircaseType == 2 && resultsFile1.results.setup.staircaseType == 2
            results.setup.staircaseType = 2;
            results.thresholdTrials.filterNum = [resultsFile1.results.thresholdTrials.filterNum, resultsFile2.results.thresholdTrials.filterNum];
        end
        file1Trials = length(resultsFile1.results.thresholdTrials.trialNum);
        file2Trials = length(resultsFile2.results.thresholdTrials.trialNum);
        results.thresholdTrials.trialNum(1:file1Trials) = resultsFile1.results.thresholdTrials.trialNum; % Load in first trials
        results.thresholdTrials.trialNum((file1Trials + 1):(file1Trials + file2Trials)) = (resultsFile1.results.thresholdTrials.trialNum(end) + 1):(file1Trials+file2Trials); % Load in second trials
        results.thresholdTrials.filters(1:file1Trials) = resultsFile1.results.thresholdTrials.filters; % Load in first trials
        results.thresholdTrials.filters((file1Trials + 1):(file1Trials + file2Trials)) = resultsFile2.results.thresholdTrials.filters; % Load in second trials
        results.thresholdTrials.response(1:file1Trials) = resultsFile1.results.thresholdTrials.response; % Load in first trials
        results.thresholdTrials.response((file1Trials + 1):(file1Trials + file2Trials)) = resultsFile2.results.thresholdTrials.response; % Load in second trials
        results.thresholdTrials.confidence(1:file1Trials) = resultsFile1.results.thresholdTrials.confidence; % Load in first trials
        results.thresholdTrials.confidence((file1Trials + 1):(file1Trials + file2Trials)) = resultsFile2.results.thresholdTrials.confidence; % Load in second trials
        results.thresholdTrials.score(1:file1Trials) = resultsFile1.results.thresholdTrials.score; % Load in first trials
        results.thresholdTrials.score((file1Trials + 1):(file1Trials + file2Trials)) = resultsFile2.results.thresholdTrials.score; % Load in second trials
        results.thresholdTrials.accuracyTotal = round(mean(results.thresholdTrials.score) * 100); % Re-calculate total accuracy
        results.thresholdTrials.adjustmentDateAndTime = datestr(now, 'yyyy_mm_dd_HHMMSS'); % Mark the date the results were manually adjusted
        outputDir = exist(fullfile('results', 'original_files')); % Check if folder for old files already exists
        if outputDir == 0
            mkdir results original_files % Make the folder if it does not exist
        elseif outputDir == 7
        end
        resultsFile1SaveName = fullfile('results', 'original_files', ['filter_task_results_', file1, '_orig', '.mat']); % Create output file and location
        resultsFile2SaveName = fullfile('results', 'original_files', ['filter_task_results_', file2, '_orig', '.mat']); % Create output file and location
        save(resultsFile1SaveName, '-struct', 'resultsFile1'); % Save original results
        save(resultsFile2SaveName, '-struct', 'resultsFile2'); % Save original results
        save(outputFile, 'results'); % Save new results
        delete (fullfile('results', ['filter_task_results_', file1, '.mat'])); % Remove old file from main results folder
        delete (fullfile('results', ['filter_task_results_', file2, '.mat'])); % Remove old file from main results folder
    else
        fprintf('\nERROR: FILTER NUMBERS DO NOT MATCH\n______________________________________\n');
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CATCH IF INCORRECT OPTION SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else
    fprintf('\nOption does not exist, please try again.\n');
end
catch
    fprintf('\n______________________________________\n')
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%