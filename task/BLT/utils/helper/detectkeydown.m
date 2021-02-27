function keyDown = detectkeydown(params, whichkey)
%DETECTKEY checks if there is an input either from the keyboard or form the
%response box 
%
% Inputs:
%   params: data storage file
%
% Outputs:
%   keyCode: vector of numbers corresponding to the pressed keys
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017


if params.keyboard == 0
    readserialbytes(params.serialPortNumber);% necessary?
    [keyCode,~] = getserialbytes(params.serialPortNumber);
    [ ~, ~, keyCode2,  ~] = KbCheck(params.deviceNumber);
    keyCode2 = find(keyCode2);
    keyCode = [keyCode, keyCode2];
    if keyCode(whichkey)
       keyDown = 1;
    end
else
    [ ~, ~, keyCode,  ~] = KbCheck(params.deviceNumber);
    if keyCode(whichkey)
       keyDown = 1;
    end
end
