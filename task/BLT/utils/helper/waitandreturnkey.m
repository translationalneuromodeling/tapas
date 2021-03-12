function [store, keyCode, ellapsed, difference] = waitandreturnkey( timeout, store ,flushevents)
%WAITANDRETURNKEY waits for a specified duration in seconds and returns
%keyCodes of all keys that were pressed during that time
%
% Syntax: WAITANDRETURNKEY(timeout, store);
% Syntax: WAITANDRETURNKEY(timeout, store, flushevents);
%
% Inputs:
%   timeout, the timeout to wait in milliseconds
%   the variable timeout has to be numeric, scalar and real
%   flushevents: logical: this function blocks the matlab event queue. In some cases
%   it's necessary to execute background task. But be aware, if this
%   variable is set to true. The task will at least use 60ms to execute.
%   store: main storage file
%
% Outputs:
%	ellapsed: the effective time passed to execute this command. Can differ
%	for some milliseconds
%   difference: the difference between timeout and the time used to execute
%   the command
%   keyCode: of the pressed key (number)
%   store: main storage file 
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017

    timeout = timeout*1000;
    ticID = tic();
    keypressed = 0;
    keypresstime = 0;
    % Input error chech
    if exist('timeout', 'var')
        if ~(isnumeric(timeout) && isscalar(timeout) && isreal(timeout))
            throw(MException('wait:timeout', 'The value timeout must be numeric, scalar and real'));
        end
    else
        throw(MException('wait:timeout', 'The input argument "timeout" is missing! Usage: wait(timeout);'));
    end
    
    if ~exist('flushevents', 'var')
        flushevents = false;        
    else
        if ~islogical(flushevents)
            try 
                flushevents = logical(flushevents);
            catch e
                e.addCause(MException('wait:flushevents', 'Keys must be numeric, real and a 1 dimensional vector'));
                rethrow(e)
            end
        end
    end
    
    timeout = timeout / 1000; % tic toc count in seconds
    ellapsed = 0;
    keyCode = [];
    
    while ellapsed <= timeout;
        ellapsed = toc(ticID);
        
        keys = detectkey(store);
        keyCode = [keyCode keys];
        wait2(10); % might cause a small delay but wont put out so many key presses
        
        if any(keyCode==store.keys.escape)
            DrawFormattedText(store.screen.window, store.text.abortText, 'center', 'center', store.screen.black);
            Screen('Flip', store.screen.window);
        
            abort = 1;
            store = logEvent('abort',store);
            save(store.safename, 'store');
            PsychPortAudio('Stop', store.tone.pahandle);
            PsychPortAudio('Close', store.tone.pahandle);
            sca
            return;
        end

    end
    
    difference = ellapsed - timeout;
end

