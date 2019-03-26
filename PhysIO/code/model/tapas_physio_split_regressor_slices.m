function slice_regressors = tapas_physio_split_regressor_slices(...
    slice_regressors_concat, nSampleSlices)
% Splits a concatenated phase regressor into slice-wise regressors, one per
% column
%
%   slice_regressors = tapas_physio_split_regressor_slices(...
%       slice_regressors_concat, nSampleSlices)
%
% IN
%   slice_regressors_concat [nSamples*nSlices, nRegressors] - matrix which
%                           per column has entries of the form:
%                           regressor value volume 1, slice 1
%                                           volume 1, slice 2 
%                                           ...
%                                           volume 1, slice nSampleSlices 
%                                           volume 2, slice 1 
%                                           volume 2, slice 2 
%                                       
% OUT
%
% EXAMPLE
%   tapas_physio_split_regressor_slices
%
%   See also

% Author: Lars Kasper
% Created: 2014-08-14
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% make multiple columns for multiple slices
nRegressors = size(slice_regressors_concat,2);
slice_regressors = [];
for iRegressor = 1:nRegressors
    slice_regressors = [slice_regressors, ...
        reshape(reshape(slice_regressors_concat(:,iRegressor),1,[]), ...
        [], nSampleSlices)];
end