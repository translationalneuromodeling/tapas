function keyCode = detectkey(params)
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
else
    [ ~, ~, keyCode,  ~] = KbCheck(params.deviceNumber);
    keyCode = find(keyCode);
end

