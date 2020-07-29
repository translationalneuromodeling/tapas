%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% BREATHING FILTER DETECTION TASK %%%%%%%%%%%%%%%%%%%%%%
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

% THIS TASK AIMS TO DETERMINE:
%   1) The number of breathing filters a participant is able to
%      discriminate from a dummy filter with 60 - 85% accuracy.
%   2) The participant's discrimination metric (d') and bias criterion (c)
%      for reporting breathing resistance, using signal detection theory.
%   3) The participant's metacognitive awareness and efficiency about their
%      performance, in the form of both average confidence over trials as
%      well as the metacognitive metric of meta-d'.

% For full information on the set up of this task please see the
% README.md file in the main task folder.

% TO RUN THIS TASK:
%   1) Copy the entire set of files to a location on your computer and add
%      folders and subfolders to your matlab path. Open the 
%      tapas_filter_detection_task.m script and set the following 
%      properties:
%           - Task type: results.setup.taskType = 1 or 2
%             1 = Yes/No task (default)
%             2 = 2IFC task
%           - Staircase type: results.setup.staircaseType = 1 or 2
%             1 = Constant (default - threshold trials at constant filter)
%             2 = Roving (filter number can change across threshold trials)
%           - Confidence scale: example (default)
%             results.setup.confidenceLower = 1
%             results.setup.confidenceUpper = 10
%   2) Navigate to the main folder (FDT) in matlab, and type 
%      tapas_filter_detection_task into the matlab terminal.
%   3) Use the supplied instructions files to explain the task requirements
%      to the participant (see FDT/participant_instructions/
%      breathing_task_instructions_{english/german}.doc).
%   4) Follow the matlab prompts to specify whether practice and
%      calibration trials should be run, and the number of trials you aim
%      to complete at threshold (minimum recommendation = 60 trials).
%   5) Turn on a low level of pink noise (required) to reduce the influence
%      of any noise cues (see README.md).
%   6) Follow the prompt instructions to complete the task. Only for the 
%      very first practice trial (a dummy) is it explicitly explained to
%      the participant whether or not they will receive the load, and then  
%      all following practice/calibration/main trials will continue on from 
%      each other with no changes in instruction, nor with any feedback
%      given.
%   7) Once the task is complete, ask the participant to fill in the
%      provided de-briefing questionnaire (see FDT/participant_debriefing/
%      debriefing_detection_{english/german}.doc).
%      This should help you to determine if any significant strategy switch
%      or learning occurred that would warrant post-hoc subject exclusion.
%
% Scripts are provided to both correct any input errors that may occur
% (see scripts/tapas_filter_detection_task_fix.m) and to run the analysis 
% (see scripts/tapas_filter_detection_analysis.m) for all subjects.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET UP OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tapas_filter_detection_task()

% Specify the task type: 1 = Yes/No task, 2 = 2IFC task
results.setup.taskType = 1; % Change this to change the task type
if results.setup.taskType == 1
    results.setup.taskDescription = 'YesNo';
elseif results.setup.taskType == 2
    results.setup.taskDescription = '2IFC';
end

% Specify the type of staircase:
% 1 = Fixed staircase, where threshold trials remain at one filter level
% 2 = Roving staircase, where threshold trials can cross filter levels
results.setup.staircaseType = 1; % Change this to change the staircase type
if results.setup.staircaseType == 1
    results.setup.staircaseDescription = 'Constant';
elseif results.setup.staircaseType == 2
    results.setup.staircaseDescription = 'Roving';
end

% Specify confidence range =
results.setup.confidenceLower = 1;
results.setup.confidenceUpper = 10;

% Set ideal accuracy band (inclusive) and error risk
if results.setup.staircaseType == 1
    results.setup.lowerBand = 0.65;
    results.setup.upperBand = 0.80;
    results.setup.errorRisk = 0.2;
elseif results.setup.staircaseType == 2
    results.setup.lowerBand = 0.70;
    results.setup.upperBand = 0.75;
    results.setup.errorRisk = 0.3;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET UP THE TASK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Close any open figures etc.
close all

% Check folder location is main FDT folder
[~,dir_name] = fileparts(pwd);
if ~strcmp(dir_name,'FDT')
   error('Not currently in main FDT folder. Please move to FDT folder and try again.');
end

% Add relevant paths
addpath('results');
addpath('figures');

% Display setup on screen
fprintf('\n_______________________________________\n\n          SET UP FILTER TASK\n_______________________________________\n');

% Prompt for PPID
results.PPID = input('PPID: ','s'); % Get subject ID
resultsFile = fullfile('results', ['filter_task_results_', results.PPID]); % Create results file name
% Check if PPID file already exists
if isfile([resultsFile, '.mat']) == 1
    error('PPID file already exists. Please choose a different PPID');
end
figureFile = fullfile('figures', ['filter_task_results_', results.PPID]); % Create figure file name

% Specify number of trials, and whether practice and calibration should be run
results.trials = input('NUMBER OF TRIALS =                   '); % Specify number of trials
practice = input('RUN PRACTICE: (Yes = 1  No = 0)?      '); % Specify whether practice trial(s) should be run
calibration = input('RUN CALIBRATION: (Yes = 1  No = 0)?   '); % Specify whether calibration trials should be run
if calibration == 0
    start_filters = input('STARTING FILTER NUMBER =              '); % Specify which filter number to use if no calibration trials will be run
end

% Specify starting filter numbers for practice trials and calibration trials
% --> Change these if required
if results.setup.taskType == 1
    results.practice.intensity = [0 7]; % Start practice at no filters and explicitly explain to participant, then move to 7 filters
    currentIntensityNum = 0;  % Start calibrations from the bottom up
elseif results.setup.taskType == 2
    results.practice.intensity = 7; % Start practice at 7 filters
    currentIntensityNum = 1;  % Start calibrations from the bottom up
end

% Specify prompt questions to ask during task
if results.setup.taskType == 1
    prompt_answer           = 'ANSWER: (Yes = 1  No = 0)             ';
elseif results.setup.taskType == 2
    prompt_answer           = 'ANSWER: (1st = 1  2nd = 2)            ';
end
prompt_confidence = sprintf('CONFIDENCE: (%d - %d)                  ', results.setup.confidenceLower, results.setup.confidenceUpper);
prompt_confirm_trial    = '-->CONFIRM TRIAL: (Yes = 1  No = 0)   ';
prompt_filters          = 'ENTER FILTERS =                       ';

% Display start on screen
fprintf('_______________________________________\n_______________________________________\n\n        FILTER DETECTION TASK\n_______________________________________\n_______________________________________\n');

% Print reminder to turn on pink noise
pinkNoise = 0;
while pinkNoise ~= 1
    try
        pinkNoise = input('PINK NOISE ON? (Yes = 1)?             '); % Prompt reminder about pink noise
        if pinkNoise ~= 1
            disp('CAUTION! Pink noise required')
        end
    catch
        disp('CAUTION! Pink noise required')
    end
end
fprintf('\nStarting task...\n_______________________________________\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN PRACTICE TRIAL(S) IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run practice trial(s) starting with a dummy and then 7 filters (specified above), can add filters if perception not reported
if practice == 1
    for n = 1:length(results.practice.intensity)
        if n == 1 && results.setup.taskType == 1
            fprintf('_______________________________________\nPRACTICE TRIAL: EXPLICIT DUMMY\n'); % Print the trial type
            fprintf('\nFILTERS = DUMMY\n'); % Print the number of filters
        else
            fprintf('_______________________________________\nPRACTICE TRIAL: LARGE LOAD\n'); % Print the trial type
            if results.setup.taskType == 1
                fprintf('\nFILTERS (%d) = PRESENT\n', results.practice.intensity(n)); % Print the number of filters
            elseif results.setup.taskType == 2
                r = randi([1, 2], 1);
                fprintf('\nFILTERS (%d) = INTERVAL %d\n', results.practice.intensity(n), r); % Print the number of filters and presentation interval
                results.practice.interval(n) = r;
            end
        end
        a = 0;
        while a < 1
            try
                response = input(prompt_answer); % Input response from participant
                if response == 1 || (response == 0 && results.setup.taskType == 1) || (response == 2 && results.setup.taskType == 2)
                    results.practice.response(n) = response;
                    b = 0;
                    while b < 1
                        try
                            confidence = input(prompt_confidence); % Ask for confidence
                            if confidence >= results.setup.confidenceLower && confidence <= results.setup.confidenceUpper
                                results.practice.confidence(n) = confidence;
                                c = 0;
                                while c < 1
                                    try
                                        confirm = input(prompt_confirm_trial); % Ask for confirmation
                                        if results.setup.taskType == 1 && n == 1 && response == 1 && confirm == 1 % If they incorrectly reported the dummy trial, repeat the trial
                                            fprintf('_______________________________________\n\n     REPEAT PRACTICE DUMMY TRIAL\n_______________________________________\n');
                                            fprintf('_______________________________________\nPRACTICE TRIAL: EXPLICIT DUMMY\n'); % Print the trial type
                                            fprintf('\nFILTERS = DUMMY\n'); % Print the number of filters
                                            c = 1; % Exit the confirm loop
                                            b = 1; % Exit the confidence loop
                                        elseif results.setup.taskType == 1 && n == 1 && response == 0 && confirm == 1 % If they correctly reported the dummy trial, move to the next trial
                                            c = 1; % Exit the confirm loop
                                            b = 1; % Exit the confidence loop
                                            a = 1; % Exit the response loop
                                        elseif (results.setup.taskType == 1 && n > 1 && response == 0 && confirm == 1) || (results.setup.taskType == 2 && response ~= results.practice.interval(n) && confirm == 1) % If they were incorrect on the filter practice trial, repeat the practice with one additional filter
                                            c = 1; % Exit the confirm loop
                                            b = 1; % Exit the confidence loop
                                            results.practice.intensity(n) = results.practice.intensity(n) + 1; % Add a filter to the practice intensity
                                            fprintf('_______________________________________\n\n     ADD FILTER (NOW = %d FILTERS)\n       + REPEAT PRACTICE TRIAL\n_______________________________________\n', results.practice.intensity(n));
                                            fprintf('_______________________________________\nPRACTICE TRIAL: LARGE LOAD\n'); % Print the trial type
                                            if results.setup.taskType == 1
                                                fprintf('\nFILTERS (%d) = PRESENT\n', results.practice.intensity(n)); % Print the number of filters
                                            elseif results.setup.taskType == 2
                                                r = randi([1, 2], 1);
                                                fprintf('\nFILTERS (%d) = INTERVAL %d\n', results.practice.intensity(n), r); % Print the number of filters and presentation interval
                                                results.practice.interval(n) = r;
                                            end
                                        elseif (results.setup.taskType == 1 && n > 1 && response == 1 && confirm == 1) || (results.setup.taskType == 2 && response == results.practice.interval(n) && confirm == 1) % If they correctly reported the filter practice trial, move to the next trial
                                            c = 1; % Exit the confirm loop
                                            b = 1; % Exit the confidence loop
                                            a = 1; % Exit the response loop     
                                        elseif confirm == 0
                                            c = 1; % Exit the confirm loop
                                            b = 1; % Exit the confidence loop
                                        end
                                    catch
                                    end
                                end
                            end
                        catch
                        end
                    end
                end
            catch
            end
        end
    end
end

% Save practice trials to results file
save(resultsFile, 'results');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN CALIBRATION TRIALS IF SPECIFIED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run calibration check to find starting point, starting with number of filters specified above
if calibration == 1
    fprintf('_______________________________________\n_______________________________________\n\n       START CALIBRATION TRIALS\n_______________________________________\n');
    for n = 1:10 % Max of 10 counted calibration trials
        results.calibration.trial(n) = n;
        results.calibration.intensity(n) = currentIntensityNum; % Start calibration with filter number specified above
        fprintf('_______________________________________\nCALIBRATION TRIAL %d\n',n); % Print the trial type and number
        if results.calibration.intensity(n) == 0
            fprintf('\nFILTERS = DUMMY\n'); % First calibration trial is a dummy --> No filters present
            results.calibration.filters(n) = 0;
        elseif results.calibration.intensity(n) > 0 && results.setup.taskType == 1
            fprintf('\nFILTERS (%d) = PRESENT\n', currentIntensityNum); % Print number of filters for all other calibration trials with Yes/No
            results.calibration.filters(n) = 1; % Filters present for all calibration trials (except dummy)
        elseif results.calibration.intensity(n) > 0 && results.setup.taskType == 2
            r = randi([1, 2], 1);
            fprintf('\nFILTERS (%d) = INTERVAL %d\n', currentIntensityNum, r); % Print the number of filters and presentation interval for all calibration trials with 2IFC
            results.calibration.interval(n) = r;
        end
        a = 0;
        while a < 1
            try
                response = input(prompt_answer); % Input response from participant
                if response == 1 || (response == 0 && results.setup.taskType == 1) || (response == 2 && results.setup.taskType == 2)
                    results.calibration.response(n) = response;
                    b = 0;
                    while b < 1
                        try
                            confidence = input(prompt_confidence); % Ask for confidence
                            if confidence >= results.setup.confidenceLower && confidence <= results.setup.confidenceUpper
                                results.calibration.confidence(n) = confidence;
                                c = 0;
                                while c < 1
                                    try
                                        confirm = input(prompt_confirm_trial); % Ask for confirmation
                                        if confirm == 1 && response == 1 && results.calibration.intensity(n) == 0 % If they report something on the dummy, repeat the dummy trial
                                            fprintf('_______________________________________\n\n          REPEAT DUMMY TRIAL\n_______________________________________\n');
                                            c = 1; % Exit the confirmation loop
                                            b = 1; % Exit the confidence loop
                                        elseif confirm == 1
                                            c = 1; % Exit the confirmation loop
                                            b = 1; % Exit the confidence loop
                                            a = 1; % Exit the response loop
                                        elseif confirm == 0
                                            c = 1; % Exit the confirmation loop
                                            b = 1; % Exit the confidence loop
                                        end
                                    catch
                                    end
                                end
                            end
                        catch
                        end
                    end
                end
            catch    
            end
        end
        
        % Score trials if 2IFC task running
        if results.setup.taskType == 2 && results.calibration.response(n) == results.calibration.interval(n)
            results.calibration.score(n) = 1;
        elseif results.setup.taskType == 2 && results.calibration.response(n) ~= results.calibration.interval(n)
            results.calibration.score(n) = 0;
        end
        
        % Initially keep changing up every step until first report yes
        if (results.setup.taskType == 1 && sum(results.calibration.response(1:n)) == 0) || (results.setup.taskType == 2 && sum(results.calibration.score(1:n)) == 0) % If no filter perception has been reported so far during calibration
            currentIntensityNum = currentIntensityNum+1; % Add a filter
            if currentIntensityNum == 1
                fprintf('_______________________________________\n\n   ADD ONE FILTER (NOW = %d FILTER)\n', currentIntensityNum);
            else
                fprintf('_______________________________________\n\n   ADD ONE FILTER (NOW = %d FILTERS)\n', currentIntensityNum);
            end
        elseif (results.setup.taskType == 1 && results.calibration.response(n-1) == 0 && results.calibration.response(n) == 1) || (results.setup.taskType == 2 && results.calibration.score(n-1) == 0 && results.calibration.score(n) == 1) % When they first report feeling the resistance, add a filter and check for yes again
            currentIntensityNum = currentIntensityNum+1; % Add a filter
            if currentIntensityNum == 1
                fprintf('_______________________________________\n\n   ADD ONE FILTER (NOW = %d FILTER)\n', currentIntensityNum);
            else
                fprintf('_______________________________________\n\n   ADD ONE FILTER (NOW = %d FILTERS)\n', currentIntensityNum);
            end
        elseif (results.setup.taskType == 1 && (results.calibration.response(n) + results.calibration.response(n-1)) == 2 && results.calibration.response(n-2) == 0) || (results.setup.taskType == 2 && (results.calibration.score(n) + results.calibration.score(n-1)) == 2 && results.calibration.score(n-2) == 0) % If two trials at ascending filters have 'yes' response
            currentIntensityNum = currentIntensityNum-1; % Remove a filter for a confirmation trial
            if currentIntensityNum == 1
                fprintf('_______________________________________\n\n  REMOVE ONE FILTER (NOW = %d FILTER)\n', currentIntensityNum);
            else
                fprintf('_______________________________________\n\n  REMOVE ONE FILTER (NOW = %d FILTERS)\n', currentIntensityNum);
            end
        elseif (results.setup.taskType == 1 && (results.calibration.response(n-1) + results.calibration.response(n-2)) == 2) || (results.setup.taskType == 2 && (results.calibration.score(n-1) + results.calibration.score(n-2)) == 2) % Once two trials at ascending filters have a 'yes' response and confirmation trial conducted
            if (results.setup.taskType == 1 && results.calibration.response(n) == 1) || (results.setup.taskType == 2 && results.calibration.score(n) == 1) % If confirmation trial at lower filter is 'yes' / correct
                break; % Filter number stays at lower intensity, calibration complete
            elseif (results.setup.taskType == 1 && results.calibration.response(n) == 0) || (results.setup.taskType == 2 && results.calibration.score(n) == 0) % If confirmation trial at lower filter is 'no' / incorrect
                currentIntensityNum = currentIntensityNum + 1; % Filter number goes up one
                break; % Calibration complete
            end
        elseif results.calibration.response(n) == 0 || results.calibration.score(n) == 0 % Otherwise, if they get one wrong
            currentIntensityNum = currentIntensityNum+1; % Add a filter
            if currentIntensityNum == 1
                fprintf('_______________________________________\n\n   ADD ONE FILTER (NOW = %d FILTER)\n', currentIntensityNum);
            else
                fprintf('_______________________________________\n\n   ADD ONE FILTER (NOW = %d FILTERS)\n', currentIntensityNum);
            end
        end
    end
    fprintf('_______________________________________\n\n         CALIBRATION COMPLETE\n');
elseif calibration == 0
    currentIntensityNum = start_filters;
end

% Save results so far
save(resultsFile, 'results');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN MAIN TASK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set running counters
n = 1;
trialCount = 1;
trialsInARow = 1;
totalTrials = 1;

% Create a dummy vector to shuffle every 10 trials for stimulus order
if results.setup.taskType == 1
    stimulus_order = [1 1 1 1 1 0 0 0 0 0];
elseif results.setup.taskType == 2
    stimulus_order = [1 1 1 1 1 2 2 2 2 2];
end

% Print starting value
if currentIntensityNum == 1
    fprintf('_______________________________________\n_______________________________________\n\n      STARTING VALUE = %d FILTER\n_______________________________________\n',currentIntensityNum);
else
    fprintf('_______________________________________\n_______________________________________\n\n      STARTING VALUE = %d FILTERS\n_______________________________________\n',currentIntensityNum);
end

% Run full set of trials, with adjustment after 10 trials as necessary
while totalTrials <= results.trials % Run this loop until specified number of trials have been completed
    
    % Specify trial number, trial count and intensity (number of filters)
    results.allTrials.trial(n) = n;
    results.allTrials.trialCount(n) = trialCount;
    results.allTrials.intensity(n) = currentIntensityNum;
    
    % Shuffle order of stimuli every 10 trials
    if n == 1 || mod(n,10) == 0 % If n is 1 or a multiple of 10
        if results.setup.taskType == 1
            results.allTrials.filters(n:n+9) = stimulus_order(randperm(length(stimulus_order))); % Initiate filter presence / absence for the next 10 trials
        elseif results.setup.taskType == 2
            results.allTrials.interval(n:n+9) = stimulus_order(randperm(length(stimulus_order))); % Initiate next filter interval for the next 10 trials
        end
    end
    
    % Present the stimulus and enter the participant's answers
    fprintf('_______________________________________\nTRIAL %d\n',totalTrials); % Print the trial type and number
    if results.setup.taskType == 1
        filters = results.allTrials.filters(n);
        if filters == 1
            fprintf('\nFILTERS (%d) = PRESENT\n', currentIntensityNum); % Print the number of filters
        elseif filters == 0
            fprintf('\nFILTERS = ABSENT\n');
        end
    elseif results.setup.taskType == 2
        interval = results.allTrials.interval(n);
        fprintf('\nFILTERS (%d) = INTERVAL %d\n', currentIntensityNum, interval); % Print the number of filters and presentation interval
    end
    a = 0;
    while a < 1
        try
            response = input(prompt_answer); % Input response from participant
            if response == 1 || response == 0 && results.setup.taskType == 1 || response == 2 && results.setup.taskType == 2 
                results.allTrials.response(n) = response;
                b = 0;
                while b < 1
                    try
                        confidence = input(prompt_confidence); % Ask for confidence
                        if confidence >= results.setup.confidenceLower && confidence <= results.setup.confidenceUpper
                            results.allTrials.confidence(n) = confidence;
                            c = 0;
                            while c < 1
                                try
                                    confirm = input(prompt_confirm_trial); % Ask for confirmation
                                    if confirm == 1
                                        c = 1; % Exit the confirmation loop
                                        b = 1; % Exit the confidence loop
                                        a = 1; % Exit the response loop
                                    elseif confirm == 0
                                        c = 1; % Exit the confirmation loop
                                        b = 1; % Exit the confidence loop
                                    end
                                catch
                                end
                            end
                        end
                    catch
                    end
                end
            end
        catch
        end
    end
    save(resultsFile, 'results');
    
    % Score trials
    if (results.setup.taskType == 1 && results.allTrials.filters(n) == results.allTrials.response(n)) || (results.setup.taskType == 2 && results.allTrials.interval(n) == results.allTrials.response(n))
        results.allTrials.score(n) = 1;
    elseif (results.setup.taskType == 1 && results.allTrials.filters(n) ~= results.allTrials.response(n)) || (results.setup.taskType == 2 && results.allTrials.interval(n) ~= results.allTrials.response(n))
        results.allTrials.score(n) = 0;
    end
    
    % Pull out threshold trials that will be used for staircase
    results.thresholdTrials.filterNum = currentIntensityNum;
    results.thresholdTrials.trialNum(trialCount) = results.allTrials.trialCount(n);
    results.thresholdTrials.response(trialCount) = results.allTrials.response(n);
    results.thresholdTrials.confidence(trialCount) = results.allTrials.confidence(n);
    results.thresholdTrials.score(trialCount) = results.allTrials.score(n);
    if results.setup.taskType == 1
        results.thresholdTrials.filters(trialCount) = results.allTrials.filters(n);
    elseif results.setup.taskType == 2
        results.thresholdTrials.interval(trialCount) = results.allTrials.interval(n);
    end
    
    % Calculate cumulative and running accuracy
    results.thresholdTrials.cumulativeAccuracy(trialCount) = round(mean(results.thresholdTrials.score(1,1:trialCount)) * 100);
    results.thresholdTrials.accuracyTotal = results.thresholdTrials.cumulativeAccuracy(trialCount);
    results.allTrials.cumulativeAccuracy(n) = results.thresholdTrials.cumulativeAccuracy(trialCount);
    results.allTrials.accuracyTotal(n) = round(mean(results.allTrials.score * 100));
    if trialCount >= 10
        results.thresholdTrials.runningAccuracy(trialCount) = round(mean(results.thresholdTrials.score((end-9):end)) * 100);
        results.allTrials.runningAccuracy(n) = results.thresholdTrials.runningAccuracy(trialCount);
    end
    
     % Plot 10-trial running accuracy and cumulative accuracy after 10 trials
    if (results.setup.staircaseType == 1 && trialCount > 10) || (results.setup.staircaseType == 2)
        if results.setup.staircaseType == 1
            x = area(results.thresholdTrials.cumulativeAccuracy, 'FaceAlpha', 0.4);
            x.FaceColor = [0.5843 0.8157 0.9882]; % Make the shaded area light blue
            hold on
            title(sprintf('\nACCURACY FOR %d FILTERS\n', currentIntensityNum))
            plot(results.thresholdTrials.trialNum, results.thresholdTrials.runningAccuracy, '-ok', 'MarkerFaceColor', 'k'); % Plot results for visualisation plot(x,y,'-o')
            legend('Total accuracy', '10-trial accuracy')
            xlim([10 results.trials])
        elseif results.setup.staircaseType == 2
            x = area(results.allTrials.accuracyTotal, 'FaceAlpha', 0.4);
            x.FaceColor = [0.5843 0.8157 0.9882]; % Make the shaded area light blue
            hold on
            title('CUMULATIVE ACCURACY')
            xlim([1 results.trials])
        end
        ylim([0 100])
        ylabel('Percentage correct')
        xlabel('Trial number')
        print(figureFile, '-dtiff');
    end
    
    % Save interim results each loop
    save(resultsFile, 'results');
    
    % Calculate probability that underlying accuracy lies within accuracy boundaries
    results.thresholdTrials.rightCount(trialCount) = sum(results.thresholdTrials.score(1:trialCount));
    results.thresholdTrials.wrongCount(trialCount) = trialCount - results.thresholdTrials.rightCount(trialCount);
    results.thresholdTrials.probBand(trialCount) = betacdf(results.setup.upperBand,2+results.thresholdTrials.rightCount(trialCount),1+results.thresholdTrials.wrongCount(trialCount)) - betacdf(results.setup.lowerBand,2+results.thresholdTrials.rightCount(trialCount),1+results.thresholdTrials.wrongCount(trialCount));
    results.allTrials.rightCount(n) = results.thresholdTrials.rightCount(trialCount);
    results.allTrials.wrongCount(n) = results.thresholdTrials.wrongCount(trialCount);
    results.allTrials.probBand(n) = results.thresholdTrials.probBand(trialCount);
    
    % Exit main loop here if number of trials is complete
    if totalTrials == results.trials
        % Clean up extra trials
        if results.setup.taskType == 1
            results.allTrials.filters = results.allTrials.filters(1:length(results.allTrials.trialCount));
        elseif results.setup.taskType == 2
            results.allTrials.interval = results.allTrials.interval(1:length(results.allTrials.trialCount));
        end
        % Clean up threshold trials if using a roving staircase
        if results.setup.staircaseType == 2
            results.runningTrials = results.thresholdTrials;
            results.thresholdTrials = results.allTrials;
            results.thresholdTrials.filterNum = results.allTrials.intensity;
            results.thresholdTrials.accuracyTotal = round(mean(results.allTrials.score * 100));
        end
        break;
    end
    
    % Suggest changes based on probable accuracy (if staircase type is constant, only when trialCount is between 5 and 30 trials)
    if trialCount >= 5 && trialsInARow >= 3 && (results.setup.staircaseType == 2 || (results.setup.staircaseType == 1 && trialCount < 30)) && ((results.setup.taskType == 1 && sum(results.allTrials.filters((n - trialsInARow + 1):n)) >= 1) || results.setup.taskType == 2) % If trialCount is between 5-30, with at least 3 trials in a row at the same filter intensity with at least one filter presented 
       if results.thresholdTrials.cumulativeAccuracy(trialCount) > 72.5 && results.thresholdTrials.probBand(trialCount) <= results.setup.errorRisk % If accuracy is above upper bound
           suggestedIntensityNum = currentIntensityNum-1; % Make it harder
           if currentIntensityNum == 1
               fprintf('_______________________________________\n_______________________________________\n\n  ACCURACY = %d PERCENT AT %d FILTER\n',results.thresholdTrials.cumulativeAccuracy(trialCount), currentIntensityNum);
           elseif currentIntensityNum > 1
               fprintf('_______________________________________\n_______________________________________\n\n  ACCURACY = %d PERCENT AT %d FILTERS\n',results.thresholdTrials.cumulativeAccuracy(trialCount), currentIntensityNum);
           end
           if suggestedIntensityNum == 0 % Avoid zero, continue on 1 filter instead
               fprintf('\n           STAY ON 1 FILTER\n_______________________________________\n');
               suggestedIntensityNum = 1;
           else
               fprintf('\n         CHANGE TO %d FILTERS?\n\n',suggestedIntensityNum);
           end
       elseif results.thresholdTrials.cumulativeAccuracy(trialCount) < 72.5 && results.thresholdTrials.probBand(trialCount) <= results.setup.errorRisk % If accuracy is less than lower bound
           suggestedIntensityNum = currentIntensityNum+1; % Make it easier
           if currentIntensityNum == 1
               fprintf('_______________________________________\n_______________________________________\n\n  ACCURACY = %d PERCENT AT %d FILTER\n',results.thresholdTrials.cumulativeAccuracy(trialCount), currentIntensityNum);
           elseif currentIntensityNum > 1
               fprintf('_______________________________________\n_______________________________________\n\n  ACCURACY = %d PERCENT AT %d FILTERS\n',results.thresholdTrials.cumulativeAccuracy(trialCount), currentIntensityNum);
           end
           fprintf('\n         CHANGE TO %d FILTERS?\n\n',suggestedIntensityNum);
       else
           suggestedIntensityNum = currentIntensityNum;
           newIntensityNum = currentIntensityNum;
       end
    else % If outside of staircase trials, keep intensity number
        suggestedIntensityNum = currentIntensityNum;
    end
    
    % If staircase type is contant, add a check every 10 trials for 30 and above, and display accuracy above and below if cumulative accuracy has fallen outside 60-85%
    if results.setup.staircaseType == 1 && trialCount >= 30 && (mod(trialCount,10) == 0) % If trial count is 30 or above, and a multiple of 10
        fprintf('_______________________________________\n_______________________________________\n\n      TRIAL COUNT = %d (%d FILTERS)\n',trialCount, currentIntensityNum);
        fprintf('\n     CURRENT ACCURACY = %d PERCENT\n',results.thresholdTrials.cumulativeAccuracy(trialCount));
        if (results.thresholdTrials.cumulativeAccuracy(trialCount) > 85) || results.thresholdTrials.cumulativeAccuracy(trialCount) < 60
            accuracyAbove = round(sum(results.allTrials.score(results.allTrials.intensity == (currentIntensityNum + 1)))/length(results.allTrials.score(results.allTrials.intensity == (currentIntensityNum + 1))) * 100);
            accuracyBelow = round(sum(results.allTrials.score(results.allTrials.intensity == (currentIntensityNum - 1)))/length(results.allTrials.score(results.allTrials.intensity == (currentIntensityNum - 1))) * 100);
            if isnan(accuracyAbove) == 1
                fprintf('\n     ACCURACY AT %d FILTERS = UNKNOWN\n',(currentIntensityNum + 1));
            else
                fprintf('\n  ACCURACY AT %d FILTERS = %d PERCENT\n',(currentIntensityNum + 1), accuracyAbove);
            end
            if isnan(accuracyBelow) == 1
                fprintf('\n    ACCURACY AT %d FILTERS = UNKNOWN\n',(currentIntensityNum - 1));
            else
                fprintf('\n  ACCURACY AT %d FILTERS = %d PERCENT\n',(currentIntensityNum - 1), accuracyBelow);
            end
        end
        fprintf('_______________________________________\n\n       CONTINUE WITH %d FILTERS?\n\n',currentIntensityNum);
        suggestedIntensityNum = currentIntensityNum;
    end

    % Confirm filter changes and reset counters if needed after 5 trials
    if trialCount >= 5 && ((suggestedIntensityNum ~= currentIntensityNum && (results.setup.staircaseType == 2 || (results.setup.staircaseType == 1 && trialCount < 30))) || (results.setup.staircaseType == 1 && trialCount >= 30 && mod(trialCount,10) == 0)) % If trialCount is greater than 5 and suggested intensity number changes
        % Confirm new filter intensity
        a = 0;
        while a < 1
            try
                prompt_confirm_filters = sprintf('CONFIRM %d FILTERS: (Yes = 1  No = 0)  ', suggestedIntensityNum);
                confirm = input(prompt_confirm_filters);
                if confirm == 1
                    newIntensityNum = suggestedIntensityNum;
                    a = 1;
                elseif confirm == 0
                    b = 0;
                    while b < 1
                        try
                            newIntensityNum = input(prompt_filters);
                            if newIntensityNum <= 10 && newIntensityNum >= 1
                                prompt_confirm_filters = sprintf('CONFIRM %d FILTERS: (Yes = 1  No = 0)  ', newIntensityNum);
                                reconfirm = input(prompt_confirm_filters);
                                if reconfirm == 1
                                    b = 1;
                                    a = 1;
                                elseif reconfirm == 0
                                    b = 0;
                                end
                            end
                        catch
                        end
                    end
                end
            catch    
            end
        end
        fprintf('_______________________________________\n');
        
        % Reset counters as needed (overall trial count reset only for constant staircase)
        if newIntensityNum == currentIntensityNum % If filter number has not changed
            trialCount = trialCount + 1; % Leave overall counter running
            if suggestedIntensityNum ~= newIntensityNum % Reset trialsInARow if suggestion has been overridden 
                trialsInARow = 1;
            else
                trialsInARow = trialsInARow + 1;
            end
        elseif newIntensityNum ~= currentIntensityNum
            currentIntensityNum = newIntensityNum; % Update filter number
            field = 'thresholdTrials';
            results = rmfield(results,field);
            results.thresholdTrials.filterNum = currentIntensityNum;
            results.thresholdTrials.trialNum = results.allTrials.trialCount(results.allTrials.intensity == results.thresholdTrials.filterNum);
            results.thresholdTrials.response = results.allTrials.response(results.allTrials.intensity == results.thresholdTrials.filterNum);
            results.thresholdTrials.confidence = results.allTrials.confidence(results.allTrials.intensity == results.thresholdTrials.filterNum);
            results.thresholdTrials.score = results.allTrials.score(results.allTrials.intensity == results.thresholdTrials.filterNum);
            if results.setup.taskType == 1
                results.thresholdTrials.filters = results.allTrials.filters(results.allTrials.intensity == results.thresholdTrials.filterNum);
            elseif results.setup.taskType == 2
                results.thresholdTrials.interval = results.allTrials.interval(results.allTrials.intensity == results.thresholdTrials.filterNum);
            end
            if isempty(results.thresholdTrials.trialNum) == 1 % If no previous intensities recorded, start trials again from 1
                trialCount = 1;
            else
                trialCount = results.thresholdTrials.trialNum(end) + 1; % Continue trial counter from previous trials at this intensity
                results.thresholdTrials.cumulativeAccuracy = results.allTrials.cumulativeAccuracy(results.allTrials.intensity == results.thresholdTrials.filterNum);
                if trialCount > 10
                    results.thresholdTrials.runningAccuracy = results.allTrials.runningAccuracy(results.allTrials.intensity == results.thresholdTrials.filterNum);
                end
            end
            trialsInARow = 1; % Re-set trials in a row counter
            if results.setup.staircaseType == 1
                close; % Close the current figure
            end
        end
    else
        trialCount = trialCount + 1;
        trialsInARow = trialsInARow + 1;
    end

    % Save interim results each loop
    save(resultsFile, 'results');
    
    % Update running total
    n = n + 1;
    
    % Update total trial counter
    if results.setup.staircaseType == 1
        totalTrials = trialCount;
    elseif results.setup.staircaseType == 2
        totalTrials = n;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE OUT FINAL RESULTS FOR ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save final results
save(resultsFile, 'results');

% Print outcome to screen
fprintf('_______________________________________\n\n         ACCURACY = %d PERCENT\n',results.thresholdTrials.accuracyTotal);
fprintf('_______________________________________\n\n         FILTER TASK COMPLETE\n_______________________________________\n');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
