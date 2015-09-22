function [dcm] = tapas_mpdcm_fmri_estimate(dcm, pars)
%% Estimates a fmri mimicking spm
%
% Input:
% dcm       -- DCM structure according to SPM
% pars      -- Structure.  Values are T, nburnin, niter, verbose, mc3
%
% Output:
% dmc       -- Structure. It contains the expected value of the model
%           parameters, The estimated free energy, using thermodynamic 
%           integration, the posterior harmonic mean, and the prior arithmetic
%           mean.
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

if nargin < 2
    pars = struct();
    pars.T = linspace(0.1, 1, 20)^5;
    pars.nburnin = 2000;
    pars.niter = 5000;
    pars.verbose = 0;
    pars.mc3 = 1;
end

% unpack

U = dcm.U;
Y = dcm.Y;
n = dcm.n;
v = dcm.v;

dcm.U = U;
dcm.Y = Y;

[ps, fe] = tapas_mpdcm_fmri_ps({dcm}, pars);


% Unpack dcm

dcm.F = fe;
dcm.ps = ps;

end
