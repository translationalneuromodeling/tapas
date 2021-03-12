function tapas_BLT_runCue(params,iTrial)
% tapas_BLT_runCue shows the Cue Stimulus
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017

% Show Cue 
if params.cue(iTrial) == 1
    Screen('DrawTexture', params.screen.window, params.text.imgcue1, [], [], 0);
elseif params.cue(iTrial) == 2
    Screen('DrawTexture', params.screen.window, params.text.imgcue2, [], [], 0);
end

Screen('Flip', params.screen.window);
tapas_BLT_logEvent('cue_on',params);

% Wait
params = wait2_escapeOption(params.dur.showCue, params);
tapas_BLT_logEvent('cue_off',params);

end

        
