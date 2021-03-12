function [ screen ] = tapas_BLT_initScreen(params)
% tapas_BLT_initScreen initilaizes the screen for the BLT experiment

% Here we call some default settings for setting up Psychtoolbox
PsychDefaultSetup(2);

% Psychtoolbox has some issues in Mac which only allows us to run this
% by skipping some default tests, we might want to change that for windows
Screen('Preference', 'SkipSyncTests', 1);

% Get the screen numbers
screens = Screen('Screens');

% Draw to the external screen if avaliable
screen.number = max(screens);

% Define black and white
screen.white = WhiteIndex(screen.number);
screen.black = BlackIndex(screen.number);
screen.grey = screen.white / 2;
screen.inc = screen.white - screen.grey;

% Open an on screen window
switch params.expMode
    case 'debug'
        rectDebug = [0 0 1300 900];
        [screen.window, screen.windowRect] = PsychImaging('OpenWindow', screen.number, screen.black, rectDebug);
    case {'task', 'train_4', 'train_6'}
        [screen.window, screen.windowRect] = PsychImaging('OpenWindow', screen.number, screen.black);
end

% Get the size of the on screen window
[screen.xpixels, screen.ypixels] = Screen('WindowSize', screen.window);

% Query the frame duration
screen.ifi = Screen('GetFlipInterval', screen.window);

% Get the centre coordinate of the window
[screen.xCenter, screen.yCenter] = RectCenter(screen.windowRect);

% Set the screen size for display
screen.fit = [screen.xCenter/2.5, screen.yCenter/3, 1.6*screen.xCenter, 1.6*screen.yCenter];

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', screen.window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Adjust the text size as specified
Screen('TextSize', screen.window, params.txt.size);


end

