function [ store, abort ] = checkEscape(store)
%CHECKESCAPE checks if the escape key was pressed and abirts the game in
%that case
%
% Outputs:
%   store: main datafile
%   abort: 1 if escape was pressed
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017


keyCode = detectkey(store);

if any(keyCode==store.keys.escape)
        DrawFormattedText(store.screen.window, store.text.abortText, 'center', 'center', store.screen.black);
        Screen('Flip', store.screen.window);
        
        abort = 1;
        store = logEvent('abort',store);
        save(store.safename, 'store');
        PsychPortAudio('Stop', store.tone.pahandle);
        % Close the audio device
        PsychPortAudio('Close', store.tone.pahandle);
        sca
        return;
else
    abort = 0;
end

end