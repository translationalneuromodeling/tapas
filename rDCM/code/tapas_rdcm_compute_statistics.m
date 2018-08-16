function [ output ] = tapas_rdcm_compute_statistics(DCM, output, options)
% Computes statistics for the posterior parameter estimates
% 
% 	Input:
%   	DCM         - model structure
%       output      - model inversion results
%       options     - estimation options
%
%   Output:
%       output      - model inversion results with statistics
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


% get rDCM posterior estimates
output.par_est = tapas_rdcm_ep2par(output.Ep);


% get true (simulations) or VBL parameters
try
    output.par_true = tapas_rdcm_ep2par(DCM.Tp);
catch
    output.par_true = tapas_rdcm_ep2par(DCM.Ep);
end


% get the true present connections
output.idx = output.par_true~=0;


% compute statistics
output.mse_n = mean((output.par_est(output.idx) - output.par_true(output.idx)).^2)/norm(output.par_true(output.idx));
output.mse   = mean((output.par_est(output.idx) - output.par_true(output.idx)).^2);
output.sign  = sum(output.par_est(output.idx).*output.par_true(output.idx)<0);


% compute signal
if ( options.compute_signal )
    output = tapas_rdcm_compute_signals(DCM, output, options);
end

end
