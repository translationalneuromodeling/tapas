function [params] = tapas_BLT_initParams(expMode)
% Initialise parameters for BLT

params.task = 'BLT';
params.expMode = expMode;

% ________________________________________________________________________%
%
%                                 Setup
% ________________________________________________________________________%

params.langMode              = 'en'; % get the language (options: 'en' or 'de')
params.predMode              = 2; % 1 for binary prediction, 2 for sliding scale prediction
params.stimAnsMode           = 1; % 1 for binary answer, 2 for sliding scale answer
params.pauseMode             = 0; % 1 for including a pause between prediction and atimulus, 0 for no pause

switch params.expMode
    case 'task'
        %HideCursor(0) % hide mouse
        params.PPID = input('PPID:','s'); % get subject ID
        
        load('tapas_BLT_cue_seq.mat'); % load predefined sequences
        params.resist        = resist;   
        params.dur.iti       = iti;
        params.cue           = cue;
        params.pairings      = pairings;
        params.valves        = 0; % specify to use the valve system
        params.CO2valve_open = -1; % specify starting value for CO2 valve
        params.co2           = [5, 10, 18, 33, 43, 54, 62, 70]; % specify which trials to give a CO2 bolus
        
        params.MRI           = 1; % turned off for testing --> for MRI turn on
        params.doIntroMain   = 1; % show the introduction slides for the main task
        params.doIntroTrain  = 0; % show the introduction slides for the training
        params.doMRI         = 0; % turned off for testing --> for MRI turn on
        params.keyboard      = 1; % 0 = use response box not keyboard, 1 = use keyboard, 2 = response box in lab
        
    case 'debug'
        params.PPID          = 'debug';
        
        % create predefined 'dummy' sequences
        params.dur.iti       = [5, 5];
        params.cue           = [2, 1];
        params.resist        = [1, 0];
        params.valves        = 0; % specify to NOT use the valve system
        params.CO2valve_open = -1; % specify starting value for CO2 valve
        params.co2           = [2]; % specify which trials to give a CO2 bolus
        
        params.MRI           = 0; 
        params.doIntroMain   = 0; % show the introduction slides for the main task
        params.doIntroTrain  = 0; % show the introduction slides for the training
        params.doMRI         = 0; % do the task in the scanner
        params.keyboard      = 1; % 0 = use response box not keyboard, 1 = use keyboard, 2 = response box in lab
       
    case 'train_4'
        %HideCursor(0) % hide mouse
        params.PPID = input('PPID:','s'); % get subject ID
        
        params.dur.iti       = [9, 8, 8, 9];
        params.cue           = [2, 1, 2, 1];
        params.resist        = [1, 0, 1, 0];
        params.valves        = 0;
        params.CO2valve_open = -1; % specify starting value for CO2 valve
        params.co2           = []; % specify which trials to give a CO2 bolus
        
        params.MRI           = 0; 
        params.doIntroTrain  = 1; % show the introduction slides for the training
        params.doIntroMain   = 0; % show the introduction slides for the main task
        params.doMRI         = 0; % do the task in the scanner
        params.keyboard      = 1; % 0 = use response box not keyboard, 1 = use keyboard, 2 = response box in lab
       
        
    case 'train_6'
        %HideCursor(0) % hide mouse
        params.PPID = input('PPID:','s'); % get subject ID
        
        params.dur.iti       = [9, 8, 8, 9, 9, 8];
        params.cue           = [2, 1, 2, 1, 1, 2];
        params.resist        = [1, 0, 1, 0, 0, 1];
        params.valves        = 0;
        params.CO2valve_open = -1; % specify starting value for CO2 valve
        params.co2           = []; % specify which trials to give a CO2 bolus
        
        params.MRI           = 0; 
        params.doIntroTrain  = 1; % show the introduction slides for the training
        params.doIntroMain   = 0; % show the introduction slides for the main task
        params.doMRI         = 0; % do the task in the scanner
        params.keyboard      = 1; % 0 = use response box not keyboard, 1 = use keyboard, 2 = response box in lab
       
end


% ________________________________________________________________________%
%
%                                PATHS
% ________________________________________________________________________%

params.path.datafolder       = fullfile('outputfiles');
params.path.datafile         = fullfile('outputfiles', [params.PPID, '_', params.expMode, '_BLT_', datestr(now, 'yyyy_mm_dd_HHMMSS')]);
params.path.imgfolder        = fullfile('images',['img_',params.langMode]);


% ________________________________________________________________________%
%
%                         RESPONSEBOX & KEYBOARD
% ________________________________________________________________________%

KbName('UnifyKeyNames');
params.scanner.mode             = 1;
params.scanner.boxport          = 2;
params.scanner.trigger          = 53;
params.keys.escape              = KbName('ESCAPE');
params.deviceNumber             = -3;

if params.keyboard == 0
    params.serialPortRate       = 19200;
    params.serialPortNumber     = 2;
    config_serial ( params.serialPortNumber , params.serialPortRate);
    params.keys.one             = 52;       % red button, r 
    params.keys.two             = 51;        % green button, g
    % blue button, b, 49;
    % yellow button, z, 50;

elseif params.keyboard == 1
    devices = PsychHID('Devices');
    for k=1:length(devices)
        dN=devices(k).usageName;
        if strcmp('Keyboard',dN)
            params.deviceNumber=devices(k).index;
        end
    end
%     params.deviceNumber = 2; % Rike's Hack if needed
    params.keys.one = KbName('LeftArrow'); %'q'
    params.keys.two = KbName('RightArrow'); %'r'
    
elseif params.keyboard == 2                 % Use this if practicing with button box in the lab
    devices = PsychHID('Devices');
    for k=1:length(devices)
        dN=devices(k).usageName;
        dT=devices(k).transport;
        if strcmp('Keyboard',dN) && strcmp('USB',dT)
            params.deviceNumber=devices(k).index;
        end
    end
    params.deviceNumber = 0; % Rike's Hack
    params.keys.one = '4$';
    params.keys.two = '3#';

end


% ________________________________________________________________________%
%
%                             VALVE CONTROL
% ________________________________________________________________________%

if params.valves == 1
    % Initialise port (if this fails, you need to install the mex-file plugin):
    config_io;
    % Set port address (you can find this in the Windows device manager):
    params.port_address = hex2dec('E010'); % - hex2dec('378'); --> Default value
end


% ________________________________________________________________________%
%
%                             DURATION (SECONDS)
% ________________________________________________________________________%

params.dur.showIntroScreen          = 2;
params.dur.showPause                = 0.5; % --> Need to test efficiency in GLM
params.dur.showpredTimeoutScreen    = 1;
params.dur.predTimeout              = 5;
params.dur.showStim                 = 5;
params.dur.stimAnsTimeout           = 3;
params.dur.rateMaxIntensity         = 10; 
params.dur.rateMaxAnxiety           = 10; 
params.dur.showrateTimeoutScreen    = 1;
params.dur.rateTimeoutIntensity     = params.dur.rateMaxIntensity - params.dur.showrateTimeoutScreen;
params.dur.rateTimeoutAnxiety       = params.dur.rateMaxAnxiety - params.dur.showrateTimeoutScreen;
params.dur.showEnd                  = 5;
params.dur.introiti                 = 10;   % Buffer for first few volumes
params.dur.CO2valve                 = 1;    % Specify how long to open the CO2 valve for (if using)
params.dur.endFieldmap              = 600;  % To Adjust
params.dur.endfinal                 = 10;


% ________________________________________________________________________%
%
%                                  TEXT
% ________________________________________________________________________%

a = 0;
while a < 1
    try
        params.cuetype = input('Enter cue type (1-4) = '); % Ask for cue-answer pairings (1 of 4 options)
        if params.cuetype <= 4 && params.cuetype >= 1
            a = 1;
        end
    catch
    end
end

if params.cuetype == 1 || params.cuetype == 3               % If the cue-answer pairing is 1 or 3
    params.answertype               = 1;                    % 'Yes' prediction is left button
    predictLeftEn                   = 'Definitely yes';     % OR 'Yes' prediction is left anchor (for scaled rating)
    predictRightEn                  = 'Definitely no';
    predictLeftDe                   = 'Definitiv ja';
    predictRightDe                  = 'Definitiv nein';
    rateLeftIntEn                   = 'Extremely difficult';
    rateRightIntEn                  = 'Not at all difficult';
    rateLeftAnxEn                   = 'Extremely anxious';
    rateRightAnxEn                  = 'Not at all anxious';
    rateLeftIntDe                   = '    Extrem schwierig';
    rateRightIntDe                  = 'Überhaupt nicht schwierig';
    rateLeftAnxDe                   = '    Extrem Ängstlich';
    rateRightAnxDe                  = 'Überhaupt nicht Ängstlich';
elseif params.cuetype == 2 || params.cuetype == 4           % If the cue-answer pairing is 2 or 4
    params.answertype               = 2;                    % 'Yes' prediction is right button
    predictLeftEn                   = 'Definitely no';      % OR 'Yes' prediction is right anchor (for scaled rating)
    predictRightEn                  = 'Definitely yes';
    predictLeftDe                   = 'Definitiv nein';
    predictRightDe                  = 'Definitiv ja';
    rateLeftIntEn                   = 'Not at all difficult';
    rateRightIntEn                  = 'Extremely difficult';
    rateLeftAnxEn                   = 'Not at all anxious';
    rateRightAnxEn                  = 'Extremely anxious';
    rateLeftIntDe                   = 'Überhaupt nicht schwierig';
    rateRightIntDe                  = '    Extrem schwierig';
    rateLeftAnxDe                   = 'Überhaupt nicht Ängstlich';
    rateRightAnxDe                  = '    Extrem Ängstlich';
end
params.nIntroTrain                  = 5;
params.nIntroMain                   = 2;
params.txt.size                     = 20;

if strcmp(params.langMode,'de')
    params.txt.Scanner              = 'Warten auf den Scanner...';
    params.txt.abort                = 'Abbruch.';
    params.txt.questionIntensity    = 'Wie schwierig war es zu atmen?';
    params.txt.anchorleftIntensity  = rateLeftIntDe;
    params.txt.anchorrightIntensity = rateRightIntDe;
    params.txt.questionAnxiety      = 'Wie besorgt waren Sie wegen Ihrer Atmung?';
    params.txt.anchorleftAnxiety    = rateLeftAnxDe;
    params.txt.anchorrightAnxiety   = rateRightAnxDe;
    params.txt.anchorleftPredict    = predictLeftDe;
    params.txt.anchorrightPredict   = predictRightDe;
elseif strcmp(params.langMode,'en')
    params.txt.Scanner              = 'Waiting for the scanner...';
    params.txt.abort                = 'Abort.';
    params.txt.questionIntensity    = 'How difficult was it to breathe?';
    params.txt.anchorleftIntensity  = rateLeftIntEn;
    params.txt.anchorrightIntensity = rateRightIntEn;
    params.txt.questionAnxiety      = 'How anxious were you about your breathing?';
    params.txt.anchorleftAnxiety    = rateLeftAnxEn;
    params.txt.anchorrightAnxiety   = rateRightAnxEn;
    params.txt.anchorleftPredict    = predictLeftEn;
    params.txt.anchorrightPredict   = predictRightEn;
end
    


