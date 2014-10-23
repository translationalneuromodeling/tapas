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
end

% unpack

U = dcm.U;
Y = dcm.Y;
n = dcm.n;
v = dcm.v;

% detrend outputs (and inputs)

Y.y = spm_detrend(Y.y);
if dcm.options.centre
    U.u = spm_detrend(U.u);
end

% check scaling of Y (enforcing a maximum change of 4%

scale = max(max((Y.y))) - min(min((Y.y)));
scale = 4/max(scale,4);
Y.y = Y.y*scale;
Y.scale = scale;

dcm.U = U;
dcm.Y = Y;

[dcm, fe] = mpdcm_fmri_ps(dcm, pars);

% Unpack dcm

dcm.F = fe;

end
