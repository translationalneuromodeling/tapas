function [otheta, oy, ollh, olpp, v] = tapas_mpdcm_fmri_sample_mh(...
    y, u, ntheta, ny, ptheta, pars, oy, otheta, ollh, olpp)
%% Draws a new sample from a Gaussian proposal distribution.
%
% Input
%   op -- Old parameters
%   ptheta -- Prior
%   htheta -- Hyperpriors
%   v -- Kernel. Two fields: s which is a scaling factor and S which is the     
%       Cholosvky decomposition of the kernel.
%
% Ouput
%   np -- New output 
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


% Compute the likelihood
nllh = pars.fllh(y, u, ntheta, ptheta, ny);

nllh = sum(nllh, 1);
nlpp = sum(pars.flpp(y, u, ntheta, ptheta), 1);

nllh(isnan(nllh)) = -inf;

v = nllh .* T + nlpp - (ollh .* T + olpp);
v = rand(size(v)) < exp(v);

ollh(v) = nllh(v);
olpp(v) = nlpp(v);
op(:, v) = np(:, v);
oy(:, v) = ny(:, v);

assert(all(-inf < ollh), 'mpdcm:fmri:ps', '-inf value in the likelihood');

end
