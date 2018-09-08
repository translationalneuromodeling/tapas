function [sstate, si] = tapas_sem_hier_get_stored_state(data, model, ...
    inference, state)
%% Get the state that should be stored 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

sstate = struct('graph', {cell(4, 1)}, 'llh', {cell(4, 1)}, 'v', []);

order = state.graph{2}.u.temperature_ordering;
sstate.graph{2} = state.graph{2}.y(:, order);

sstate.llh{1} = state.llh{1}(:, order);

sstate.v = state.v(:, order);

if state.nsample < inference.nburnin
    si = mod(state.nsample, inference.ndiag) + 1;
else
    si = state.nsample - inference.nburnin + 1;
end

end
