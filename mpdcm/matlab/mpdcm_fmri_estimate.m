function [dcm] = mpdcm_fmri_estimate(dcm, pars)
%% Estimates a fmri mimicking spm
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 2
    pars = struct();
    pars.T = linspace(0.1, 1, 20)^5;
    pars.nburnin = 2000;
    pars.niter = 5000;
    pars.verbose = 0;
end

% unpack

U = dcm.U;
Y = dcm.Y;
n = dcm.n;
v = dcm.v;

dcm.U = U;
dcm.Y = Y;

[ps, fe] = mpdcm_fmri_ps({dcm}, pars);


% Unpack dcm

dcm.F = fe;
dcm.ps = ps;

end
