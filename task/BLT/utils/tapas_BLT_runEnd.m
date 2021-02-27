function tapas_BLT_runEnd(params)
% tapas_BLT_runEnd ends the task and stores and close everything

if params.doMRI == 1
    % Show preliminary end screen for the duration of the field map 
    Screen('DrawTexture', params.screen.window, params.text.endfieldmap, [], params.screen.fit, 0);
    Screen('Flip', params.screen.window);

    % Wait
    params = wait2_escapeOption(params.dur.endFieldmap, params);
end

% Show final screen
Screen('DrawTexture', params.screen.window, params.text.endfinal, [], params.screen.fit, 0);
Screen('Flip', params.screen.window);

% Wait
params = wait2_escapeOption(params.dur.endfinal, params);

tapas_BLT_logEvent('end',params);

% Clear the screen & remove paths
sca;
