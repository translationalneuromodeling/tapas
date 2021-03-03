% ________________________________________________________________________%
%
% Breathing Learning Task (BLT)
%
% Originally coded: May 2018 by F.H. Petzschner and O.K. Harrison
% Contact: O.K. Harrison (faull@biomed.ee.ethz.ch)
%
% ________________________________________________________________________%
% Info: This is the laboratory version of the BLT
% This task runs a cue-learning sequence that is coupled to a manually
% applied breathing stimulus.
% ________________________________________________________________________%

% Clear the workspace and the screen
sca;
close all;
clearvars;

% Check folder location is main BLT folder
[~,dir_name] = fileparts(pwd);
if ~strcmp(dir_name,'BLT')
   error('Not currently in main BLT folder. Please move to BLT folder and try again.');
end

% Add relevant paths
addpath('utils');
addpath('images');
addpath(fullfile('utils','helper'));
addpath('outputfiles');
KbName('UnifyKeyNames');


%% _______________________________________________________________________%
%
%
%                                  INIT
%
% ________________________________________________________________________%

% determine game mode
expMode          = input('Options (task,train_4,train_6,debug):','s');

% init experimental parameters
params           = tapas_BLT_initParams(expMode);

% init screen
params.screen    = tapas_BLT_initScreen(params);

% init visual stimuli
params.text      = tapas_BLT_initVisuals(params);

% init data file
tapas_BLT_initDatafile(params);

% init cogent & parallel ports if in MRI scanner
if params.doMRI == 1
    tapas_BLT_initCogent(params);
    tapas_BLT_initParallelPorts(params);
else
    load(params.path.datafile)
    data.events.start_protocol = GetSecs();
    save(params.path.datafile, 'data', 'params');
end


%% _______________________________________________________________________%
%
%
%                            INTRO & TRAINING
%
% ________________________________________________________________________%

if params.doIntroMain == 1
    tapas_BLT_runIntroMain(params);
end

if params.doIntroTrain == 1
    tapas_BLT_runIntroTrain(params);
end


%% _______________________________________________________________________%
%
%
%                                MAIN LOOP
%
% ________________________________________________________________________% 

ticTime = tic(); % get the time
data.events.start_sequence = GetSecs();
save(params.path.datafile, 'data', 'params');

% run buffer ITI if required
tapas_BLT_runBuffer(params);

% RUN MAIN LOOP
nTrial = length(params.cue);

for iTrial = 1:nTrial
    
    % Print the trial number to the screen
    fprintf('TRIAL NUMBER = %d\n', iTrial);
    
    % run prediction
    if params.predMode == 1
        tapas_BLT_runPrediction(params, iTrial);
    elseif params.predMode == 2
        tapas_BLT_runPredictionScale(params, iTrial);
    end
    
    % Can run a pause here if required
    if params.pauseMode == 1
        tapas_BLT_runPause(params);
    end
    
    % run Stimulus
    tapas_BLT_runStimulus(params, iTrial);
    
    % run stimulus question
    if params.stimAnsMode == 1
        tapas_BLT_runStimAnswer(params);
    elseif params.stimAnsMode == 2
        tapas_BLT_runRating(params, 1);
    end
    
    % run ITI
    tapas_BLT_runITI(params, iTrial);
        
end


%% _______________________________________________________________________%
%
%
%                          FINAL ANXIETY RATING
%
% ________________________________________________________________________% 

% run end (almost)
tapas_BLT_runEnd_almost(params);

% run Rating for intensity (= 1) if wasn't asked throughout task
if params.stimAnsMode == 1
    tapas_BLT_runRating(params, 1);
end

% run Rating for anxiety (= 2)
tapas_BLT_runRating(params, 2);


%% _______________________________________________________________________%
%
%
%                                FINISH
%
% ________________________________________________________________________%

% show end if needed
if params.doIntroMain == 1
    tapas_BLT_runEnd(params);
else
    Screen('CloseAll');
end
