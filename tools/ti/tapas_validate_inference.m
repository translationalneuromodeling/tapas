function tapas_validate_inference(inference)
%% Check whether the input is complient with the interface. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isstruct(inference)
    error('tapas:validate:mcmc:inference', ...
        'inference is not a structure');
end

isfunc = @(x) isa(x, 'function_handle');

validate_field(inference, 'estimate_method', isfunc);
validate_field(inference, 'initialize_states', isfunc);
validate_field(inference, 'initialize_state', isfunc);
validate_field(inference, 'sampling_methods', @iscell);

for i = 1:numel(inference.sampling_methods)
    if ~ isfunc(inference.sampling_methods{i})
        error('tapas:validate:mcmc:inference', ...
            'inference.sampling_methods{%s} is no a function', i);
    end
end

validate_field(inference, 'metasampling_methods', @iscell);

for i = 1:numel(inference.metasampling_methods)
    if ~ isfunc(inference.metasampling_methods{i})
        error('tapas:validate:mcmc:inference', ...
            'inference.metasampling_methods{%s} is no a function', i);
    end
end


validate_field(inference, 'get_stored_state', isfunc);
validate_field(inference, 'prepare_posterior', isfunc);
validate_field(inference, 'niter', @isnumeric);
validate_field(inference, 'nburnin', @isnumeric);


end

function validate_field(inference, field, validate_type)
%% Validates a field

if ~isfield(inference, field)
    error('tapas:validate:mcmc:inference', ...
        'inference lacks field %s', field);
end

if ~ validate_type(getfield(inference, field))
    error('tapas:validate:mcmc:inference', ...
        'inference.%s is of incorrect type', field)
end 

end
