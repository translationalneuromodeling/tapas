function [ h ] = tapas_rdcm_get_convolution_bm(options)
% [ h ] = tapas_rdcm_get_convolution_bm(options)
% 
% Creates a fixed hemodynamic response function (HRF) by convolving a single 
% event (impulse) with the standard Balloon model from DCM.
%
%   Input:
%       options     - options structure with relevant information
% 
%   Output:
%       h           - hemodynamic response function (HRF)
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

% get the DCM
DCM = options.DCM;

% number of input data points
N = size(DCM.U.u,1);

% check if full convolution should be generated
if isfield(options,'conv_full') & options.conv_full
    r_dt = 1;
else
    r_dt = N/options.conv_length;
end

% construct a dummy DCM
DCM.a    = -1;
DCM.b    = 0;
DCM.c    = 1;
DCM.d    = zeros(1,1,0);
DCM.Ep   = tapas_rdcm_empty_par(DCM);
DCM.Ep.A = DCM.a;
DCM.Ep.C = DCM.c*16;

% setting input of the fake DCM
DCM.U.u = zeros(size(DCM.U.u,1),1);
DCM.U.u(1:r_dt,:) = 1;

% setting additional parameters of the fake DCM
DCM.Y.dt = DCM.U.dt;
DCM.n    = 1;
DCM.v    = N;
DCM.ns   = N;

% create the HRF from the dummy DCM
DCM = tapas_dcm_euler_make_indices(DCM);
y   = tapas_dcm_euler_gen(DCM, DCM.Ep);

% sample the HRF at the sampling rate of the data
h = y(1:r_dt:end,1);
h = h;

end
