function store = tapas_BLT_initParallelPorts(store)
% tapas_BLT_initParallelPorts function that waits for input on the parallel port from the MRI scanner
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017

% show wait for scanner screen
DrawFormattedText(store.screen.window, store.txt.Scanner, 'center', 'center', store.screen.white);
Screen('Flip', store.screen.window);

% wait for the scanner and save the starting time
clearserialbytes(store.scanner.boxport);
[~, t] = waitserialbyte(store.scanner.boxport, inf, store.scanner.trigger);
store.MRIstart=GetSecs;
store.startScan.Serial = t(1);
clearserialbytes(store.scanner.boxport);

% after receiving the scanner trigger wait for 10 s
Screen('Flip', store.screen.window);
store = wait2_escapeOption(10, store);

end