function [ DCM ] = tapas_rdcm_generate(DCM, options, SNR)
% Generates synthetic fMRI data under a given signal to noise ratio (SNR) 
% with the fixed hemodynamic convolution kernel
% 
% 	Input:
%   	DCM         - either model structure or a file name
%       options     - estimation options
%       SNR         - signal to noise ratio
%
%   Output:
%       DCM         - model structure with generated synthetic time series
%
% 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2018 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% Setting parameters
if ~isempty(options) && options.y_dt
    DCM.Y.dt = options.y_dt;
end
r_dt   = DCM.Y.dt/DCM.U.dt;
[N, ~] = size(full(DCM.U.u));
nr     = size(DCM.a,1);

% specify the array for the data
y = zeros(N, nr);

% get the hemodynamic response function (HRF)
h = options.h;

% set the driving input
if ( isfield(options,'scale_u') )
    DCM.Tp.C = DCM.Tp.C*16;
end

% Getting neuronal signal (x)
DCM.U.u = [DCM.U.u; DCM.U.u; DCM.U.u];
DCM     = tapas_dcm_euler_make_indices(DCM);
[~, x]  = tapas_dcm_euler_gen(DCM, DCM.Tp);
DCM.U.u = DCM.U.u(1:N,:);

% set the driving input
if ( isfield(options,'scale_u') )
    DCM.Tp.C = DCM.Tp.C/16;
end

% Convolving neuronal signal with HRF
for i = 1:nr
    tmp = ifft(fft(x(:,i)).*fft([h; zeros(N*3-length(h),1)]));
    y(:,i) = tmp(N+1:2*N);
end

% Sampling
y = y(1:r_dt:end,:);

% Adding noise
eps = randn(size(y))*diag(std(y)/SNR);
y_noise = y + eps;

% Saving the generated data
DCM.Y.y = y_noise;
DCM.y = y;
DCM.x = x(N+1:2*N);

end
