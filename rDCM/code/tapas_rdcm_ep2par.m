function [ par ] = tapas_rdcm_ep2par(Ep)
% [ par ] = tapas_rdcm_ep2par(Ep)
% 
% Vectorizes Ep structure
% 
%   Input:
%   	Ep          - parameter structure
%
%   Output:
%   	par         - vectorized version of parameters
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


% create vectorized version of parameters
par = [Ep.A(:); Ep.B(:); Ep.C(:)];

end
