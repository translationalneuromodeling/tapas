function [sstate, si] = tapas_h2gf_get_stored_state(data, model, ...
    inference, state)
%% Get the state that should be stored 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

sstate = struct('graph', {cell(4, 1)}, 'llh', {cell(4, 1)}, 'v', []);

sstate.graph{2} = state.graph{2}.y;
sstate.graph{3} = state.graph{3}.y;

sstate.llh{1} = state.llh{1};

sstate.v = state.v;

if state.nsample < inference.nburnin
    si = mod(state.nsample, inference.ndiag) + 1;
else
    si = state.nsample - inference.nburnin + 1;
end

if si == inference.niter
    sstate.graph{1}.T = state.T{1};
end

end

