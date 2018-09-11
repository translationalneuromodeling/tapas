function [ptheta] = tapas_hgf_ptheta()
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

ptheta = struct();

ptheta.name = 'hgf';
% Independent generation of the states.
ptheta.llh = @tapas_ti_llh_state;
ptheta.lpp = @tapas_ti_lpp;

ptheta.method_llh = @tapas_hgf_llh;
ptheta.method_state = @tapas_hgf_gen_state;
ptheta.method_lpp = @tapas_hgf_lpp;

ptheta.prepare_ptheta = @tapas_hgf_prepare_ptheta;
ptheta.ptrans = @(x) x;

end

