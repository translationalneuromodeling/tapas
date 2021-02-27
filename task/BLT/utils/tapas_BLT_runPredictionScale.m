function tapas_BLT_runPredictionScale(params, iTrial)
% tapas_BLT_runPrediction is showing the prediction screen with scaled prediction between 0 and 100

screenw = 2*params.screen.xCenter;
screenh = 2*params.screen.yCenter;

% horizontal line
horizonLineh        = screenh*0.008;
horizonLinew        = screenw*0.6;
horizonLineCoord    = [0 0 horizonLinew horizonLineh];
horizonyPos         = params.screen.yCenter+params.screen.yCenter/3;
horizonLineRect     = CenterRectOnPointd(horizonLineCoord, params.screen.xCenter,horizonyPos);

% vertical line
verticalLineh       = screenh*0.1;
verticalLinew       = horizonLineh*2;
if params.keyboard == 0 || params.keyboard == 2     % Increase width for button box in behavioural lab and scanner
    verticalLinew   = verticalLinew*4;
end
verticalLineCoord   = [0 0 verticalLinew verticalLineh];
verticalLineCoordBold = [0 0 verticalLinew verticalLineh];
verticalxPos        = params.screen.xCenter;
verticalLineRect    = CenterRectOnPointd(verticalLineCoord, verticalxPos, horizonyPos);
minxpos             = screenw/2 - (horizonLinew )/2;
maxxpos             = screenw/2 + (horizonLinew )/2;
verticalStepsize    = verticalLinew;

% middle line
middleLineCoord     = [0 0 verticalLinew verticalLineh/3];
middleLineRect      = CenterRectOnPointd(middleLineCoord, params.screen.xCenter, horizonyPos);

% end line left
endLineLeftCoord    = [0 0 verticalLinew verticalLineh/3];
verticalLineLeftRect= CenterRectOnPointd(endLineLeftCoord, minxpos, horizonyPos);

% end line right
endLineRightCoord    = [0 0 verticalLinew verticalLineh/3];
verticalLineRightRect= CenterRectOnPointd(endLineRightCoord, maxxpos, horizonyPos);

% anchor text
anchoryPos          = horizonyPos - verticalLineh;
anchorxPosLeft      = minxpos - 1.2*verticalLineh;
anchorxPosRight     = maxxpos - 1.2*verticalLineh;

% Show cue + prediction combo
if params.cue(iTrial) == 1
    Screen('DrawTexture', params.screen.window, params.text.imgcue1Scale, [], params.screen.fit, 0);
elseif params.cue(iTrial) == 2
    Screen('DrawTexture', params.screen.window, params.text.imgcue2Scale, [], params.screen.fit, 0);
end

% Write the question      
DrawFormattedText(params.screen.window, params.txt.anchorleftPredict, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
DrawFormattedText(params.screen.window, params.txt.anchorrightPredict, anchorxPosRight , anchoryPos, [1 1 1], 50);

% Draw the sliding scale
Screen('FillRect', params.screen.window, [1 1 1],horizonLineRect)
Screen('FillRect', params.screen.window, [1 1 1],middleLineRect)
Screen('FillRect', params.screen.window, [1 1 1],verticalLineRect)
Screen('FillRect', params.screen.window, [1 1 1],verticalLineLeftRect)
Screen('FillRect', params.screen.window, [1 1 1],verticalLineRightRect)

Screen('Flip', params.screen.window);  
tapas_BLT_logEvent('cue_pred_on',params);

tic()
response = 0;
waiting = 1;

while waiting
    ellapsed = toc();
    if ellapsed > params.dur.predTimeout
        waiting = 0;
    end

    if params.keyboard == 1 || params.keyboard == 2
        keyDownLeft = checkkeydown(params, params.keys.one);
        keyDownRight = checkkeydown(params, params.keys.two);
    elseif params.keyboard == 0
        [keyCode, ~] = waitserialbyte(params.serialPortNumber, 100, [ params.keys.one params.keys.two ]);
        if keyCode > 0
            if keyCode(end) == params.keys.one
                keyDownLeft = 1;
                keyDownRight = 0;
            elseif keyCode(end) == params.keys.two
                keyDownRight = 1;
                keyDownLeft = 0;
            end
        else
            keyDownRight = 0;
            keyDownLeft = 0;
        end
    end

    if keyDownLeft
        response = 1;
        verticalxPos = verticalxPos - verticalStepsize;
        if verticalxPos < minxpos
            verticalxPos = minxpos;
        end
        verticalLineRect  = CenterRectOnPointd(verticalLineCoord, verticalxPos ,params.screen.yCenter+params.screen.yCenter/3);
    elseif keyDownRight
        response = 1;
        verticalxPos = verticalxPos + verticalStepsize;
        if verticalxPos > maxxpos
            verticalxPos = maxxpos;
        end
        verticalLineRect  = CenterRectOnPointd(verticalLineCoord, verticalxPos ,params.screen.yCenter+params.screen.yCenter/3);
    end

    % Show cue + prediction combo
    if params.cue(iTrial) == 1
        Screen('DrawTexture', params.screen.window, params.text.imgcue1Scale, [], params.screen.fit, 0);
    elseif params.cue(iTrial) == 2
        Screen('DrawTexture', params.screen.window, params.text.imgcue2Scale, [], params.screen.fit, 0);
    end

    % Write the question      
    DrawFormattedText(params.screen.window, params.txt.anchorleftPredict, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
    DrawFormattedText(params.screen.window, params.txt.anchorrightPredict, anchorxPosRight , anchoryPos, [1 1 1], 50);

    % Draw the sliding scale
    Screen('FillRect', params.screen.window, [1 1 1],horizonLineRect)
    Screen('FillRect', params.screen.window, [1 1 1],middleLineRect)
    Screen('FillRect', params.screen.window, [1 1 1],verticalLineRect)
    Screen('FillRect', params.screen.window, [1 1 1],verticalLineLeftRect)
    Screen('FillRect', params.screen.window, [1 1 1],verticalLineRightRect)
    Screen('Flip', params.screen.window);  

    if params.keyboard == 0 || params.keyboard == 1 % Don't include this when using button box in the lab --> too fast
        wait2_escapeOption(0.02, params);
    elseif params.keyboard == 2
        wait2_escapeOption(0.18, params); % Slow down button box in the lab
    end
end

% Final Outcome
if response
    verticalLineRectBold = CenterRectOnPointd(verticalLineCoordBold,verticalxPos,horizonyPos);

    % Show cue + prediction combo
    if params.cue(iTrial) == 1
        Screen('DrawTexture', params.screen.window, params.text.imgcue1Scale, [], params.screen.fit, 0);
    elseif params.cue(iTrial) == 2
        Screen('DrawTexture', params.screen.window, params.text.imgcue2Scale, [], params.screen.fit, 0);
    end
    
    % Write the question      
    DrawFormattedText(params.screen.window, params.txt.anchorleftPredict, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
    DrawFormattedText(params.screen.window, params.txt.anchorrightPredict, anchorxPosRight , anchoryPos, [1 1 1], 50);

    % Draw the sliding scale with different colour for final response position
    Screen('FillRect', params.screen.window, [1 1 1],horizonLineRect)
    Screen('FillRect', params.screen.window, [1 1 1],middleLineRect)
    Screen('FillRect', params.screen.window, [0 0 1],verticalLineRectBold)
    Screen('FillRect', params.screen.window, [1 1 1],verticalLineLeftRect)
    Screen('FillRect', params.screen.window, [1 1 1],verticalLineRightRect)
    
    % Record the answer
    if params.answertype == 1
        ratinganswer = 100 - ((verticalxPos-minxpos)/horizonLinew)*100; % Transform so definitely Yes = 100
    elseif params.answertype == 2
        ratinganswer  = ((verticalxPos-minxpos)/horizonLinew)*100; % Definitely Yes = 100
    end
else 
    ratinganswer = NaN;
    Screen('DrawTexture', params.screen.window, params.text.imgratetimeout, [], params.screen.fit, 0);
end

Screen('Flip', params.screen.window);

if response
    % Display warning if responded but no decision was made
    if verticalxPos == params.screen.xCenter
        ratinganswer = 999;
        Screen('DrawTexture', params.screen.window, params.text.imgprednoresponse, [], params.screen.fit, 0);
        Screen('Flip', params.screen.window);
    end
end

tapas_BLT_logData('pred_answer', params, ratinganswer);
wait2_escapeOption(params.dur.showpredTimeoutScreen, params);
tapas_BLT_logEvent('cue_pred_off',params);

end
