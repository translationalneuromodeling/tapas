function [ DCM ] = tapas_dcm_euler_make_indices(DCM)
% [ DCM ] = tapas_dcm_euler_make_indices(DCM)
% 
% Get the indices for the Euler integration
% 
%   Input:
%   	DCM         - model structure
%
%   Output:
%       DCM         - model structure with indices
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


% length of input
L = size(DCM.U.u,1);

% number of regions
nr = size(DCM.a, 1);

% check whether there is a delay specified
if ~isfield('DCM','delay')
    DCM.delay = ones(nr,1);
end

% create index array
Indices = repmat(1 : L, 1, 1);

% get the indices that coincide wiht the data timepoints
idx = zeros(1, L);
idx(DCM.delay:floor(DCM.Y.dt/DCM.U.dt):end) = 1;

% asign those timings
Indices = Indices(idx>0);
DCM.M.idx = Indices;

end