function keyDown = checkkeydown(params, whichkey)
% WAITFORKEY - this functions waits for a key press of a specific key indicated by the
% keynumber in whichkey
% Code: F.Petzschner 19. April 2017
% Last change: F.Petzschner 19. April 2017
keyDown = 0;
if params.keyboard == 0
    % wait for response box key
    clearserialbytes(params.serialPortNumber);
    [~, keyCode] = waitserialbyte(params.serialPortNumber,inf, whichkey);
else
    % wait for keyboard key
    [~, ~, keyCode] = KbCheck(params.deviceNumber);
    if keyCode(whichkey)
        keyDown = 1;
    end 
end