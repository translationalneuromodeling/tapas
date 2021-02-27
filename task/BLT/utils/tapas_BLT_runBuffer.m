function tapas_BLT_runBuffer(params)
% tapas_BLT_runBuffer runs an initial fixation cross

Screen('DrawTexture', params.screen.window, params.text.imgiti, [], params.screen.fit, 0);
Screen('Flip', params.screen.window);
wait2_escapeOption(params.dur.introiti , params);
