function keyCode = waitforkey(params, whichkey)
% WAITFORKEY - this functions waits for a key press of a specific key indicated by the
% keynumber in whichkey
% Code: F.Petzschner 19. April 2017
% Last change: F.Petzschner 19. April 2017

if params.keyboard == 0
    % wait for response box key
    clearserialbytes(params.serialPortNumber);
    [~, keyCode] = waitserialbyte(params.serialPortNumber,inf, whichkey);
else
    while 1
        % wait for keyboard key
        [~, keyCode, ~] = KbWait(params.deviceNumber);
        if keyCode(whichkey)
            break
        end
    end
    
end