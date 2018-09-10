function tapas_validate_model(model)
%% Verify that an object is complient with the specification. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~ isstruct(model)
    error('tapas:validate:model', 'model is not a structure')
end

if ~ isfield(model, 'graph')
    error('tapas:validate:model', 'model does not have field graph');
end

if ~ iscell(model.graph)
    error('tapas:validate:model', 'model.graph is not a cell');
end

for i = 1:numel(model.graph)
    if ~ isstruct(model.graph{i})
        error('tapas:validate:model', ...
            'model.graph{%s} is not a struct', i);
    end
    if ~ isfield(model.graph{i}, 'llh')
        error('tapas:validate:model', ...
            'model.graph{%d} does not have field llh', i);
    end
    if ~ isfield(model.graph{i}, 'htheta')
        error('tapas:validate:model', ...
            'model.graph{%d} does not have field htheta', i);
    end
end


end

