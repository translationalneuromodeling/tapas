function [dummy, nx] = tapas_h2gf_gen_state(data, theta, ptheta)
%% Generate the trace of the states.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

ptheta = ptheta.hgf;

[prc, ~] = tapas_hgf_get_theta(theta, ptheta);

prc_fun = ptheta.c_prc.prc_fun;

r = ptheta;
r.y = data.y;
r.u = data.u;
r.ign = data.ign;
r.irr = data.irr;

[dummy, nx] = prc_fun(r, prc, 'trans');


end

