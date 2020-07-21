function [posterior] = tapas_mcmc_estimate(data, model, inference)
%% Estimate a model specified. 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

nfuncs = numel(inference.sampling_methods);
[states] = inference.initialize_states(data, model, inference);
[state] = inference.initialize_state(data, model, inference);

[sstates, si] = inference.get_stored_state(data, model, inference, state);
if si > 0 
    states{si} = sstate;
end

for i = 1:inference.niter + inference.nburnin + 1
    for j = 1:nfuncs
         method = inference.sampling_methods{j};
         state = method(data, model, inference, state);
         
         % Store the state according to the method
         ns = ( i - 1 ) * nfuncs + j;
         state.nsample = ns;
         % Get a state in the format to be stored
         [sstate, si] = inference.get_stored_state(data, model, inference, ...
             state);
         % If there is an index store the state
         if si > 0
             states{si} = sstate;
         end

         % Diagnostics
    end
end

posterior = inference.prepare_posterior(data, model, inference, states);

end

