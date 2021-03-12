function tapas_BLT_runRating(params, rating)
% tapas_BLT_runRating shows a rating scale via two bars to indicate a number between 0 and 100.
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017

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
verticalLineRect    = CenterRectOnPointd(verticalLineCoord,verticalxPos ,horizonyPos);
minxpos             = screenw/2 - (horizonLinew )/2;
maxxpos             = screenw/2 + (horizonLinew )/2;
verticalStepsize    = verticalLinew;

% end line left
endLineLeftCoord    = [0 0 verticalLinew verticalLineh/3];
verticalLineLeftRect= CenterRectOnPointd(endLineLeftCoord,minxpos ,horizonyPos);

% end line right
endLineRightCoord    = [0 0 verticalLinew verticalLineh/3];
verticalLineRightRect= CenterRectOnPointd(endLineRightCoord,maxxpos ,horizonyPos);

% anchor text
anchoryPos          = horizonyPos - verticalLineh;
anchorxPosLeft      = minxpos - 1.5*verticalLineh;
anchorxPosRight     = maxxpos - 1.5*verticalLineh;

% Show the question depending on type specified
if rating == 1
    DrawFormattedText(params.screen.window, params.txt.questionIntensity, 'center' , params.screen.yCenter/1.5, [1 1 1], 50);      
    DrawFormattedText(params.screen.window, params.txt.anchorleftIntensity, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
    DrawFormattedText(params.screen.window, params.txt.anchorrightIntensity, anchorxPosRight , anchoryPos, [1 1 1], 50);
elseif rating == 2
    DrawFormattedText(params.screen.window, params.txt.questionAnxiety, 'center' , params.screen.yCenter/1.5, [1 1 1], 50);      
    DrawFormattedText(params.screen.window, params.txt.anchorleftAnxiety, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
    DrawFormattedText(params.screen.window, params.txt.anchorrightAnxiety, anchorxPosRight , anchoryPos, [1 1 1], 50);
end
Screen('FillRect', params.screen.window, [1 1 1],horizonLineRect)
Screen('FillRect', params.screen.window, [1 1 1],verticalLineRect)
Screen('FillRect', params.screen.window, [1 1 1],verticalLineLeftRect)
Screen('FillRect', params.screen.window, [1 1 1],verticalLineRightRect)

Screen('Flip', params.screen.window);  
tapas_BLT_logEvent('rating_on',params);

tic()
response = 0;
waiting = 1;

while waiting
    ellapsed = toc();
    if rating == 1
       if ellapsed > params.dur.rateTimeoutIntensity
       waiting = 0;
       end
    elseif rating == 2
       if ellapsed > params.dur.rateTimeoutAnxiety
       waiting = 0;
       end
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

    % Show the question depending on type specified
    if rating == 1
        DrawFormattedText(params.screen.window, params.txt.questionIntensity, 'center' , params.screen.yCenter/1.5, [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorleftIntensity, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorrightIntensity, anchorxPosRight , anchoryPos, [1 1 1], 50);
    elseif rating == 2
        DrawFormattedText(params.screen.window, params.txt.questionAnxiety, 'center' , params.screen.yCenter/1.5, [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorleftAnxiety, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorrightAnxiety, anchorxPosRight , anchoryPos, [1 1 1], 50);
    end
    Screen('FillRect', params.screen.window, [1 1 1],horizonLineRect)
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
    verticalLineRectBold = CenterRectOnPointd(verticalLineCoordBold,verticalxPos ,horizonyPos);
    if rating == 1
        DrawFormattedText(params.screen.window, params.txt.questionIntensity, 'center' , params.screen.yCenter/1.5, [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorleftIntensity, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorrightIntensity, anchorxPosRight , anchoryPos, [1 1 1], 50);
    elseif rating == 2
        DrawFormattedText(params.screen.window, params.txt.questionAnxiety, 'center' , params.screen.yCenter/1.5, [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorleftAnxiety, anchorxPosLeft , anchoryPos , [1 1 1], 50);      
        DrawFormattedText(params.screen.window, params.txt.anchorrightAnxiety, anchorxPosRight , anchoryPos, [1 1 1], 50);
    end
    Screen('FillRect', params.screen.window, [1 1 1],horizonLineRect)
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
if rating == 1
    tapas_BLT_logData('rate_answer_diff', params, ratinganswer);
elseif rating == 2
    tapas_BLT_logData('rate_answer_anx', params, ratinganswer);
end
wait2_escapeOption(params.dur.showrateTimeoutScreen, params);
tapas_BLT_logEvent('rating_off',params);

