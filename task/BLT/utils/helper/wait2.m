function [ellapsed, difference] = wait2( timeout, flushevents)
%WAIT waits for a specified duration in milliseconds
%
% Syntax: WAIT(timeout);
% Syntax: WAIT(timeout, flushevents);
%
% Inputs:
%   timeout, the timeout to wait in milliseconds
%   the variable timeout has to be numeric, scalar and real
%   flushevents: logical: this function blocks the matlab event queue. In some cases
%   it's necessary to execute background task. But be aware, if this
%   variable is set to true. The task will at least use 60ms to execute.
%
% Outputs:
%	ellapsed: the effective time passed to execute this command. Can differ
%	for some milliseconds
%   difference: the difference between timeout and the time used to execute
%   the command
%
% About and Copyright of this function
%   Author: Adrian Etter
%   E-Mail: adrian.etter@econ.uzh.ch
%   © SNS-Lab,
%   University of Zurich
%   Version 1.0 2012/September/4
%   Last revision: 2012/September/4
%   -finished & released

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
        if flushevents == true
            drawnow();
        end
    end
    
    difference = ellapsed - timeout;
end

