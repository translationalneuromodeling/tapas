function [y] = tapas_mpdcm_fmri_int(u, theta, ptheta, sloppy)
%% Integrates the system of differential equations specified by the input.
%
% Input:
% u         -- Cell array of experimental input
% theta     -- Cell array of model parameteres
% ptheta    -- Structure. Model priors or constants
% sloppy    -- Scalar. If true don't check the input. 
%
% Ouput:
% y         -- Cell array of predicted signals.
%
% If the input is not compliant, it's very likely that a segmentation fault
% happens and that matlab closes. It should only be used once the input has
% been check at least once and changes to them are done via well tested
% functions.
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

if nargin < 4
    sloppy = 0;
end

if ~isfield(ptheta, 'arch')
    ptheta.arch = 'cpu';
end

switch ptheta.arch
case 'cpu'
    y = tapas_mpdcm_fmri_int_host(u, theta, ptheta, sloppy);
case 'gpu'
    y = tapas_mpdcm_fmri_int_cuda(u, theta, ptheta, sloppy);
case 'gpu_asynchronous'
    y = tapas_mpdcm_fmri_int_cuda_asynchronous(u, theta, ptheta, sloppy);
otherwise
    error('tapas:mpdcm:fmri:int:input', 'Unkwon arch');
end


end
