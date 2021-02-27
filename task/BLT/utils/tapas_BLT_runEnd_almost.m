function tapas_BLT_runEnd_almost(params)
% tapas_BLT_runEnd_almost shows a screen once the task is finished but further
% measures are still needed

% Show Cue 
Screen('DrawTexture', params.screen.window, params.text.endfieldmap, [], params.screen.fit, 0);
Screen('Flip', params.screen.window);

% Wait
params = wait2_escapeOption(3, params);
