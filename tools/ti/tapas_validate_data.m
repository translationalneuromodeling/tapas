function tapas_validate_data(data)
%% Check that data complies with the inferface.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isstruct(data)
    error('tapas:validate:data', 'data is not a structure')
end

if ~isfield(data, 'y')
    error('tapas:validate:data', 'data lacks field y');
end

if ~isfield(data, 'u')
    error('tapas:validate:data', 'data lacks field u');
end


end

