function [y] = tapas_mpdcm_fmri_spm_int(Ep, M, U)
%% Interface to spm so that mpdcm can be used by spm_nlsi_GN. 
%
% Input
%   Ep      Matrices A, B, C, and D
%   M       Model parameters
%   U       Input U.
%
% Output
%   y       Predicted signal
%

% aponteeduardo@gmail.com
% copyright (C) 2016
%

if ~isfield(M, 'pars');
    M.pars = struct();
end

pars = M.pars;

if ~isfield(pars, 'arch')
    pars.arch = 'cpu';
end

if ~isfield(pars, 'integ')
    pars.integ = 'rk4';
end

theta = tapas_mpdcm_fmri_init_theta(Ep);
[u, udt] = tapas_mpdcm_fmri_init_u(U);

ptheta = init_ptheta();

ptheta.arch = pars.arch;
ptheta.integ = pars.integ;

ptheta.udt = udt;
ptheta.dyu = M.ns / size(u, 2);


y = tapas_mpdcm_fmri_int({u}, {theta}, ptheta);

y = y{1};

end % tapas_mpdcm_fmri_spm_int 


function [ptheta] = init_ptheta()
%% Generates the structure with ptheta

ptheta = struct('dt', [], 'dyu', [], 'udt', []);

ptheta.dyu = 1;
ptheta.dt = 1;

end
