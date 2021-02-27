function tapas_BLT_runStimAnswer(params)
% tapas_BLT_runStimAnswer is showing the stimulus question / answer screen

% Show stimulus question
Screen('DrawTexture', params.screen.window, params.text.imgstimanswer, [], params.screen.fit, 0);

% Flip and log
Screen('Flip', params.screen.window);
tapas_BLT_logEvent('stim_ans_on',params);

tic()
ellapsed = 0;
responded = 0;

while ellapsed < params.dur.stimAnsTimeout
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
        Screen('DrawTexture', params.screen.window, params.text.imgstimanswerSL, [], params.screen.fit, 0);
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
        Screen('DrawTexture', params.screen.window, params.text.imgstimanswerSR, [], params.screen.fit, 0);
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
tapas_BLT_logData('stim_answer_rt', params, rt);
tapas_BLT_logData('stim_answer', params, answer);
tapas_BLT_logEvent('stim_ans_off',params);

end
