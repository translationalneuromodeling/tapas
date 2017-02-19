function [PROSACCADE, ANTISACCADE] = tapas_action_codes()
%% Returns the codes for pro and antisaccades.
%
% Input
%
% Output
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

codes = c_action_codes();

PROSACCADE = codes(1);
ANTISACCADE = codes(2);

end % tapas_action_codes 

