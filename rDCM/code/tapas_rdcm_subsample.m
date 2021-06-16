function [ y ] = tapas_rdcm_subsample(Y, r_dt)
% [ y ] = tapas_rdcm_subsample(Y, r_dt)
% 
% Subsamples signal Y (in frequency domain) with a rate r_dt
% 
%   Input:
%       Y       - original signal
%       r_dt    - time step (delta_t)
%
%   Output:
%       y       - subsampled signal
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


% dimensionality of data
[N, dim] = size(Y);
p = factor(r_dt);
y = zeros(N/r_dt,dim);

for i = 1:dim
    y_tmp = Y(:,i);
    for j = 1:length(p)
        y_tmp = decimate(y_tmp, p(j),12); % decimating (subsampling) u
    end
    y(:,i) = y_tmp;
end

% subsample the data
y = Y(1:r_dt:end,:)/r_dt;

end
