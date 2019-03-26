function [state] = tapas_mcmc_meta_diagnostics(data, model, ...
    inference, state, states, si)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%

if si < 1
  return;
end

if ~mod(si, inference.ndiag)
    v = zeros(size(state.v));
    tsi = si - inference.ndiag;
    for i = 1:inference.ndiag
        v = v + states{tsi + i}.v;
    end
    fprintf(1, '-------------\n')
    fprintf(1, 'Sample: %d\n', state.nsample + 1);
    fprintf(1, '-------------\n')
    fprintf(1, 'Accept rate: ');
    fprintf(1, '%0.2f, ', mean(v, 1)/inference.ndiag);
    fprintf(1, '\n');
    fprintf(1, 'Likelihood: ');
    fprintf(1, '%0.2f, ', sum(states{si}.llh{1}, 1));
    fprintf(1, '\n');
    if isfield(state, 'time')
        toc(state.time)
        state.time = tic;
    end
end

end

