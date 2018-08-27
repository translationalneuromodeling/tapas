function [ output ] = tapas_rdcm_compute_signals(DCM, output, options)
% [ output ] = tapas_rdcm_compute_signals(DCM, output, options)
% 
% Computes true and predicted signals
% 
% 	Input:
%   	DCM         - model structure
%       output      - model inversion results
%       options     - estimation options
%
%   Output:
%   	output      - model inversion results with model fits
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


% true or measured signal
output.signal.y_source = DCM.Y.y(:);

% true (deterministic) signal / VBL signal
if strcmp(options.type,'s')
    
    % noise-free signal
    DCM = tapas_rdcm_generate(DCM, options, Inf);
    output.signal.y_clean = DCM.Y.y(:);
    
    % compute the MSE of the noisy data
    output.residuals.y_mse_clean = mean((output.signal.y_source - output.signal.y_clean).^2);
    
else
    
    % VBL predicted signal
    if ( isfield(DCM,'y') )
        output.signal.y_pred_vl     = DCM.y(:);
        output.residuals.y_mse_vl   = mean((output.signal.y_source - output.signal.y_pred_vl).^2);
    elseif ( isfield(DCM,'Ep') )
        DCM                         = tapas_rdcm_spm_dcm_generate(DCM,Inf);
        output.signal.y_pred_vl     = DCM.Y.y(:);
        output.residuals.y_mse_vl   = mean((output.signal.y_source - output.signal.y_pred_vl).^2);
    end
end

% adding the constant baseline
DCM.U.u(:,end+1) = ones(size(DCM.U.u,1),1);

% get rDCM parameters
DCM.Tp   = output.Ep;

% no baseline for simulated data
if ( strcmp(options.type,'s') )
    DCM.Tp.baseline = zeros(size(DCM.Tp.C,1),1);
end

% include the baseline
DCM.Tp.C            = [DCM.Tp.C, DCM.Tp.baseline];
DCM.Tp.B(:,:,end+1) = DCM.Tp.B(:,:,1);
DCM.Y.dt            = DCM.U.dt;

% posterior probability (for sparse rDCM)
if ( isfield(output,'Ip') )
    DCM.Tp.A = DCM.Tp.A .* output.Ip.A;
    DCM.Tp.C = DCM.Tp.C .* [output.Ip.C, ones(size(DCM.Tp.baseline))];
end

% generate predicted signal (tapas_rdcm_generate)
DCM = tapas_rdcm_generate(DCM, options, Inf);
output.signal.y_pred_rdcm = DCM.Y.y(:);

% compute the MSE of predicted signal
output.residuals.y_mse_rdcm = mean((output.signal.y_source - output.signal.y_pred_rdcm).^2);
output.residuals.R_rdcm     = output.signal.y_source - output.signal.y_pred_rdcm;

% rdcm predicted signal (tapas_rdcm_spm_dcm_generate)
if options.compute_signal_spm
    DCM.Ep = DCM.Tp;
    if ( strcmp(options.type,'r') && isfield(options,'convolution') )
        DCM.Ep.C = DCM.Ep.C/16;
    else
        DCM.Ep.C = DCM.Ep.C;
    end
    DCM.Y.dt = options.y_dt;
    DCM.v = size(DCM.Y.y,1);
    DCM = tapas_rdcm_spm_dcm_generate(DCM,Inf);
    output.signal.y_pred_rdcm_spm = DCM.Y.y(:);
    
    % compute the MSE of predicted signal
    output.residuals.y_mse_rdcm_spm = mean((output.signal.y_source - output.signal.y_pred_rdcm_spm).^2);
    output.residuals.R_rdcm_spm     = output.signal.y_source - output.signal.y_pred_rdcm_spm;
end

% asign the region names
if ( isfield(DCM.Y,'name') )
    output.signal.name = DCM.Y.name;
end

end