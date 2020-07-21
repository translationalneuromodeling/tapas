function [nx] = tapas_hgf_gen_state(y, x, u, theta, ptheta)
%% Generate the trace of the states.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

[prc, ~] = tapas_hgf_get_theta(theta, ptheta);

prc_fun = ptheta.prc_fun;
ptrans_prc = ptheta.obs_fun;
try
    [dummy, nx] = prc_fun(ptheta.r, prc, 'trans');
catch err
    c = strsplit(err.identifier, ':');
    if strcmp(c{1}, 'tapas')
        nx = nan;
        return
    else
        rethrow(err)
    end
end


end

