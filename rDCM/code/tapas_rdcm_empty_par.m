function [ Ep ] = tapas_rdcm_empty_par(DCM)
% [ Ep ] = tapas_rdcm_empty_par(DCM)
% 
% Creates an empty valid parameter structure for a given DCM
% 
%   Input:
%   	DCM         - model structure
%
%   Output:
%   	Ep          - parameter structure
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


% number of regions and inputs
[nr, nu] = size(DCM.c);

% empty parameter structure
Ep.A        = zeros(nr,nr);
Ep.B        = zeros(nr,nr,nu);
Ep.C        = zeros(nr,nu);
Ep.D        = zeros(nr,nr,0);
Ep.transit  = zeros(nr,1);
Ep.decay    = zeros(nr,1);
Ep.epsilon  = 0;

end
