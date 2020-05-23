function [posterior] = tapas_mcmc_blocked_estimate(data, model, inference)
%% Estimate a model specified. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nfuncs = numel(inference.sampling_methods);
nmeta = numel(inference.metasampling_methods);

[states] = inference.initialize_states(data, model, inference);
[state] = inference.initialize_state(data, model, inference);

tapas_validate_state(state);

[sstate, si] = inference.get_stored_state(data, model, inference, state);

if si > 0 
    states{si} = sstate;
end

for i = 2:inference.niter + inference.nburnin
    for j = 1:nfuncs
         method = inference.sampling_methods{j};
         state = method(data, model, inference, state);         
    end
    % Get a state in the format to be stored
    [sstate, si] = inference.get_stored_state(data, model, inference, ...
        state);
    % If there is an index store the state
    if si > 0
        states{si} = sstate;
    end

    for j = 1:nmeta
        [state] = inference.metasampling_methods{j}(...
            data, method, inference, state, states, si);
    end
    
end

posterior = inference.prepare_posterior(data, model, inference, states);

end

