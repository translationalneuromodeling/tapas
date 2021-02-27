function tapas_BLT_logEvent(eventName, params)
% tapas_BLT_logEvent logs timing of the event specified in eventName
%
% Inputs:
%   store: data storage file
%   eventName: string with event name
%
% Outputs:
%	store: data storage file
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017
% 

load(params.path.datafile);
eventTime               = GetSecs() - data.events.start_sequence;
data.events.(eventName) = [data.events.(eventName), eventTime];
save(params.path.datafile, 'data', 'params');
end
