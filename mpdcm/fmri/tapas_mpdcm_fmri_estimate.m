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
end


if ~isfield(pars, 'T')
    pars.T = linspace(0.0001, 1, 60)^5;
end

if ~isfield(pars, 'verb')
    pars.verb = 0;
end

if ~isfield(pars, 'mc3i')
    pars.mc3i = ceil(numel(pars.T) * 0.7);
end

if ~isfield(pars, 'diagi')
    pars.diagi = 200;
end

if ~isfield(pars, 'integ')
    pars.integ = 'rk4';
end

if ~isfield(pars, 'arch')
    pars.arch = 'cpu';
end

if ~isfield(pars, 'dt')
    pars.dt = 1;
end

if ~isfield(pars, 'rinit')
    pars.rinit = 0;
end

if ~isfield(pars, 'samples')
    pars.samples = 1;
end

if ~isfield(pars, 'fllh')
    pars.fllh =  @tapas_mpdcm_fmri_llh;
end

if ~isfield(pars, 'flpp')
    pars.flpp =  @tapas_mpdcm_fmri_lpp;
end

if ~isfield(pars, 'gibbs')
    pars.gibbs = @tapas_mpdcm_fmri_sample_gibbs;
end

if ~isfield(pars, 'prepare_ptheta')
    pars.prepare_ptheta = @tapas_mpdcm_fmri_prepare_ptheta;
end

if ~isfield(pars, 'algorithm')
    pars.algorithm = @tapas_mpdcm_fmri_ps;
end

if ~isfield(pars, 'nms')
    pars.nms = 20;
end

if numel(pars.T) == 1
    pars.mc3i = 0;
end

dcm = tapas_mpdcm_fmri_prepare_options(dcm); 

algorithm = pars.algorithm;

[ps, fe] = algorithm({dcm}, pars);

% Unpack dcm

dcm.F = fe;
dcm.ps = ps;
dcm.pars = pars;

end

