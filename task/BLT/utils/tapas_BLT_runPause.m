function tapas_BLT_runPause(params)
% tapas_BLT_runPause shows a pause (fication cross)
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017

% Show fixation cross for pause
Screen('DrawTexture', params.screen.window, params.text.imgiti, [], params.screen.fit, 0);
Screen('Flip', params.screen.window);
tapas_BLT_logEvent('pause_on',params);

% Wait
params = wait2_escapeOption(params.dur.showPause, params);
tapas_BLT_logEvent('pause_off',params);

end

        
