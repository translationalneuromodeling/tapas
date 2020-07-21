function [htheta] = tapas_hgf_htheta(ptheta)
%% Initilize a kernel for the hgf.
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%


hprc = 200.0 * eye(numel(ptheta.r.c_prc.priormus));
hobs = 200.0 * eye(numel(ptheta.r.c_obs.priormus));

htheta.pk = blkdiag(hprc, hobs);


end

