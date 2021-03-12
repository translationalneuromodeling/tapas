function tapas_BLT_runPrediction(params,iTrial)
% tapas_BLT_runPrediction is showing the prediction screen

% Show cue + prediction combo
if params.cue(iTrial) == 1
    Screen('DrawTexture', params.screen.window, params.text.imgcue1, [], params.screen.fit, 0);
elseif params.cue(iTrial) == 2
    Screen('DrawTexture', params.screen.window, params.text.imgcue2, [], params.screen.fit, 0);
end

% Flip and log
Screen('Flip', params.screen.window);
tapas_BLT_logEvent('cue_pred_on',params);

tic()
ellapsed = 0;
responded = 0;

while ellapsed < params.dur.predTimeout
    ellapsed = toc();
    keyCode = detectkey(params);
    
    if keyCode == params.keys.one
        responded = 1;
        rt      = ellapsed;
        
        % Code the answer
        if params.answertype == 1
            answer  = 1; % Yes
        elseif params.answertype == 2
            answer  = 0; % No
        end
        
        % Draw and flip the screen
        if params.cue(iTrial) == 1
            Screen('DrawTexture', params.screen.window, params.text.imgcue1SL, [], params.screen.fit, 0);
        elseif params.cue(iTrial) == 2
            Screen('DrawTexture', params.screen.window, params.text.imgcue2SL, [], params.screen.fit, 0);
        end
        Screen('Flip', params.screen.window);
        
    elseif keyCode == params.keys.two
        responded = 1;
        rt      = ellapsed;
        
        % Code the answer
        if params.answertype == 1
            answer  = 0; % No
        elseif params.answertype == 2
            answer  = 1; % Yes
        end
        
        % Draw and flip the screen
        if params.cue(iTrial) == 1
            Screen('DrawTexture', params.screen.window, params.text.imgcue1SR, [], params.screen.fit, 0);
        elseif params.cue(iTrial) == 2
            Screen('DrawTexture', params.screen.window, params.text.imgcue2SR, [], params.screen.fit, 0);
        end
        Screen('Flip', params.screen.window);
    end
end
    
% Display warning if too slow
if responded == 0
    Screen('DrawTexture', params.screen.window, params.text.imgpredtimeout , [], params.screen.fit, 0);
    Screen('Flip', params.screen.window);
    wait2_escapeOption(params.dur.showpredTimeoutScreen, params);
    answer = NaN;
    rt = NaN;
end

% Log the data
tapas_BLT_logData('pred_rt', params, rt);
tapas_BLT_logData('pred_answer', params, answer);
tapas_BLT_logEvent('cue_pred_off',params);

end
