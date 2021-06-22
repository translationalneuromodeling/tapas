function [ Y ] = tapas_rdcm_reduce_zeros(X, Y)
% [ Y ] = tapas_rdcm_reduce_zeros(X, Y)
% 
% If there are more zero-valued frequencies than informative ones,
% subsamples those frequencies to balance dataset
% 
%   Input:
%   	X           - design matrix (predictors)
%       Y           - data
%
%   Output:
%       Y           - balanced data
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


% get all indices
idx = 1:size(Y,1);

% data
data = sum(abs([Y X]),2);

% zero frequencies
idx_0 = idx(data == 0);

% number of zero frequencies
n0 = sum(data==0);

% number of non-zero and non NaN frequencies
n1 = sum(data>0);

% balance the data if there are too many zeros
if ( n0 > n1 )
    idx_del = [zeros(1,n1) ones(1,n0-n1)];
    idx_del = idx_del(randperm(n0))>0;
    Y(idx_0(idx_del),:) = NaN;
end

end
