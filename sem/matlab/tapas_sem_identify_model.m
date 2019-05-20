function [model, param] = tapas_sem_identify_model(fllh)
%% Identifies the model from the likelihood function.
%
% Input
%       fllh        -- Funcion handle with the likelihood of the model
% Output
%       model       -- String. Name of the model (seria or prosa).
%       param       -- String. Name of the parametric distribution.
%

% aponteeduardo@gmail.com
% copyright (C) 2019
%

parts = split(func2str(fllh), '_');

if numel(parts) < 4
    error('tapas:sem:identify_model', ...
        'Name of the function couldn''t be identified');
end

model = parts{2};

switch model
    case 'seria'
        % Go on
    case 'prosa'
        % Go on
    otherwise
        error('tapas:sem:identify_model', ...
            'Name of the function couldn''t be identified');
end

param = parts{4};

switch param
    case 'gamma'
        % Go on
    case 'mixedgamma'
        % Go on
    case 'invgamma'
        % Go on
    case 'lognorm'
        % go on
    case 'wald'
        % go on
    case 'later'
        % go on
    otherwise
        error('tapas:sem:identify_model', ...
            'Name of the function couldn''t be identified');
end

end
