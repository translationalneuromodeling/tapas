% Stop playback
PsychPortAudio('Stop', audios.pahandle);
% Close the audio device
PsychPortAudio('Close', audios.pahandle);
% Clear the screen
sca;