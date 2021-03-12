function [params, ellapsed, difference, abort] = wait2_escapeOption( timeout, params ,flushevents)
%WAIT2_ESCAPEOPTION waits for a specified duration checkes at
% the same time if the abort (ESCAPE) key was pressed
%
% Syntax: WAIT2_ESCAPEOPTION(timeout, params);
% Syntax: WAIT2_ESCAPEOPTION(timeout, params, flushevents);
%
% Inputs:
%   timeout, the timeout to wait in milliseconds
%   the variable timeout has to be numeric, scalar and real
%   flushevents: logical: this function blocks the matlab event queue. In some cases
%   it's necessary to execute background task. But be aware, if this
%   variable is set to true. The task will at least use 60ms to execute.
%   params: main storage file
%
% Outputs:
%	ellapsed: the effective time passed to execute this command. Can differ
%	for some milliseconds
%   difference: the difference between timeout and the time used to execute
%   the command
%   abort: returns if game was aborted 
%   params: main storage file 
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017


    timeout = timeout*1000;
    ticID = tic();
    
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
    
    while ellapsed <= timeout;
        ellapsed = toc(ticID);
        [ params, abort ] = checkEscape(params);
        if flushevents == true
            drawnow();
        end
    end
    
    difference = ellapsed - timeout;
end

