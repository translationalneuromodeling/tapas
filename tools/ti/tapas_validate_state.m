function tapas_validate_state(state)
%% Checks that a state complies with the interface.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~ isstruct(state)
    error('tapas:validate:state', 'state is not a structure');
end

if ~ isfield(state, 'graph')
    error('tapas:validate:state', 'state lacks field graph');
end

if ~ isfield(state, 'llh')
    error('tapas:validate:state', 'state lacks field llh');
end

if ~ isfield(state, 'nsample')
    error('tapas:validate:state', 'state lacks field nsample');
end


for i = 1:numel(state.graph)
    if ~isstruct(state.graph{i})
        error('tapas:validate:state', ...
            'state.graph{%d} should be a struct', i);
    end
    if ~isfield(state.graph{i}, 'y')
        error('tapas:validate:state', ...
            'state.graph{%d} lacks field y', i);
    end
    if ~isfield(state.graph{i}, 'u')
        error('tapas:validate:state', ...
            'state.graph{%d} lacks field u', i);
    end
end

end

