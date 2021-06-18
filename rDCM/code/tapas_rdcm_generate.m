function [ DCM ] = tapas_rdcm_generate(DCM, options, SNR)
% [ DCM ] = tapas_rdcm_generate(DCM, options, SNR)
% 
% Generates synthetic fMRI data under a given signal to noise ratio (SNR) 
% with the fixed hemodynamic convolution kernel
% 
%   Input:
%   	DCM         - model structure
%       options     - estimation options
%       SNR         - signal to noise ratio
%
%   Output:
%       DCM         - model structure with generated synthetic time series
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2021 Translational Neuromodeling Unit
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


% compile source code of integrator
tapas_rdcm_compile()

% Setting parameters
if ~isempty(options) && options.y_dt
    DCM.Y.dt = options.y_dt;
end
r_dt   = DCM.Y.dt/DCM.U.dt;
[N, ~] = size(full(DCM.U.u));
nr     = size(DCM.a,1);

% specify the array for the data
y = zeros(N, nr);

% generate fixed hemodynamic response function (HRF)
if ( ~isfield(options,'h') || numel(options.h) ~= size(DCM.U.u,1) )
    options.DCM         = DCM;
    options.conv_dt     = DCM.U.dt;
    options.conv_length = size(DCM.U.u,1);
    options.conv_full   = 'true';
    options.h           = tapas_rdcm_get_convolution_bm(options);
end

% get the hemodynamic response function (HRF)
h = options.h;

% Getting neuronal signal (x)
DCM.U.u = [DCM.U.u; DCM.U.u; DCM.U.u];
DCM     = tapas_dcm_euler_make_indices(DCM);
[~, x]  = tapas_dcm_euler_gen(DCM, DCM.Tp);
DCM.U.u = DCM.U.u(1:N,:);

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
