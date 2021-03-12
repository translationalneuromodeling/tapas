function close_all_ports

load tone
PsychPortAudio('Stop', tone.pahandle);
% Close the audio device
PsychPortAudio('Close', tone.pahandle);
% Clear the screen
sca;
