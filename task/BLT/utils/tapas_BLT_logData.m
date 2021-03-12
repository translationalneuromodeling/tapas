function tapas_BLT_logData(dataName, params, input)
% tapas_BLT_logData logs data specific to the staircase paradigm
%data
%
% Inputs:
%   params: data storage file
%   eventName: string with event name
%
% Outputs:
%	params: data storage file
%

% Matlab time
load(params.path.datafile);
data.(dataName) = [data.(dataName), input];
save(params.path.datafile, 'data', 'params');
end

