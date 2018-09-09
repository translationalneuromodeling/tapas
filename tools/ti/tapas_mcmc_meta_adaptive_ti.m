function [state] = tapas_mcmc_meta_adaptive_ti(data, model, inference, ...
    state, states, si)
%% 
%
% Input
%       
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2016
%


if state.nsample > inference.nburnin || mod(si, inference.ndiag)
    return;
end

% Number of subjects and chains
[ns, nc] = size(states{6}.llh{1});

% No real adapation is possible
if nc <= 4
    return;
end


llh = zeros(nc, inference.ndiag);
g = [];
for i = 1:inference.ndiag
    llh(:, i) = sum(states{i}.llh{1}, 1);
    if any(isnan(llh(:, i))) || any(abs(llh(:, i)) == inf);
        g = [g i];
    end
end

% Remove nans if any
llh(:, g) = [];
t = state.T{1}(1, :)';

% Maybe something is not working with the first derivatives.
% Let's try to find a workable schedule
for i = 1:nc - 4
    try
        [~, ~, nt] = tapas_genpath(t, llh, i, 'hermite');
        % It worked
        break
    catch
        % Ok keep going
    end
end

% It is a mess and we could not estimate anything
if i == nc - 4
    return
else
    nt = [t(1:i - 1); nt];
end

% Update the temperature schedule
state.T{1} = repmat(nt', ns, 1);
hold on; plot(mean(llh, 2));
end

