function tapas_BLT_runStimulus(params, iTrial)
% tapas_BLT_runStimulus shows the screen when the breathing stimulus is present 

% Show Stimulus 
Screen('DrawTexture', params.screen.window, params.text.imgstimulation, [], params.screen.fit, 0);
Screen('Flip', params.screen.window);
tapas_BLT_logEvent('stim_on',params);

% Close air valve if resistance is on (if using valves)
if params.valves == 1 && params.resist(iTrial) == 1
    outp(params.port_address, 1);
elseif params.valves == 1 && params.resist(iTrial) == 0
    outp(params.port_address, 1);
    pause(0.07);
    outp(params.port_address, 0);
end

% Wait
params = wait2_escapeOption(params.dur.showStim, params);
tapas_BLT_logEvent('stim_off',params);

% Re-open air valve (if using)
if params.valves == 1 && params.resist(iTrial) == 1
    outp(params.port_address, 0);
end

end

        
