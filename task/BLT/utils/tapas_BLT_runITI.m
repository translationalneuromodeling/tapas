function tapas_BLT_runITI(params, iTrial)
% tapas_BLT_runITI shows the Fixation Cross
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017

% Show Cue 
Screen('DrawTexture', params.screen.window, params.text.imgiti, [], params.screen.fit, 0);
Screen('Flip', params.screen.window);
tapas_BLT_logEvent('iti_on',params);

% Open CO2 valve if specified (and using valves)
if params.valves == 1 && (sum(ismember(params.co2, iTrial)) == 1)
    outp(params.port_address, 2);
    params.CO2valve_open = GetSecs;
end

% Wait
params = wait2_escapeOption(params.dur.iti(iTrial), params);
tapas_BLT_logEvent('iti_off',params);

% Ensure valves are closed
if params.valves == 1
    outp(params.port_address, 0);
end

end

