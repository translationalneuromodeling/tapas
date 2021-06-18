function [ output ] = tapas_rdcm_compute_signals(DCM, output, options)
% [ output ] = tapas_rdcm_compute_signals(DCM, output, options)
% 
% Computes true and predicted signals
% 
%   Input:
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


% true or measured signal
output.signal.y_source = DCM.Y.y(:);

% true (deterministic) signal / VBL signal
if strcmp(options.type,'s')
    
    % if signal should be computed
    if ( options.compute_signal )
    
        % noise-free signal
        DCMs = tapas_rdcm_generate(DCM, options, Inf);
        output.signal.y_clean = DCMs.Y.y(:);

        % compute the MSE of the noisy data
        output.residuals.y_mse_clean = mean((output.signal.y_source - output.signal.y_clean).^2);
        
    end
    
else
    
    % VBL predicted signal
    if ( isfield(DCM,'y') )
        output.signal.y_pred_vl     = DCM.y(:);
        output.residuals.y_mse_vl   = mean((output.signal.y_source - output.signal.y_pred_vl).^2);
    else
        output.residuals.y_mse_vl   = [];
    end
end


% store true or measured temporal derivative (in frequency domain)
yd_source_fft                           = output.temp.yd_source_fft;
yd_source_fft(~isfinite(yd_source_fft)) = 0;
output.signal.yd_source_fft             = yd_source_fft(:);


% if signal in time domain should be computed
if ( options.compute_signal )

    % get rDCM parameters
    DCM.Tp   = output.Ep;

    % no baseline for simulated data
    if ( strcmp(options.type,'s') )
        DCM.Tp.baseline = zeros(size(DCM.Tp.C,1),1);
    end

    % increase sampling rate
    r_dt     = DCM.Y.dt/DCM.U.dt;
    DCM.Y.dt = DCM.U.dt;

    % posterior probability (for sparse rDCM)
    if ( isfield(output,'Ip') )
        DCM.Tp.A = DCM.Tp.A .* output.Ip.A;
        DCM.Tp.C = DCM.Tp.C .* output.Ip.C;
    end


    % generate predicted signal (tapas_rdcm_generate)
    DCM_rDCM = tapas_rdcm_generate(DCM, options, Inf);

    % add the confounds to predicted time series
    for t = 1:size(DCM.Y.y,1)
        for r = 1:size(DCM.Y.y,2)
            DCM_rDCM.Y.y(t,r) = DCM_rDCM.Y.y(t,r) + DCM_rDCM.Tp.baseline(r,:) * DCM_rDCM.U.X0((1+r_dt*(t-1)),:)';
        end
    end

    % turn into vector
    output.signal.y_pred_rdcm = DCM_rDCM.Y.y(:);
    
end


% store predicted temporal derivative (in frequency domain)
yd_pred_rdcm_fft                = output.temp.yd_pred_rdcm_fft;
output.signal.yd_pred_rdcm_fft  = yd_pred_rdcm_fft(:);

% remove the temp structure
output = rmfield(output,'temp'); 


% asign the region names
if ( isfield(DCM.Y,'name') )
    output.signal.name = DCM.Y.name;
end


% compute the MSE of predicted signal
if ( options.compute_signal )
    output.residuals.y_mse_rdcm     = mean((output.signal.y_source - output.signal.y_pred_rdcm).^2);
    output.residuals.R_rdcm         = output.signal.y_source - output.signal.y_pred_rdcm;
end


% store the driving inputs
output.inputs.u     = DCM.U.u;
output.inputs.name  = DCM.U.name;
output.inputs.dt    = DCM.U.dt;

end