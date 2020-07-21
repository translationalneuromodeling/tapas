function rrf = tapas_physio_rrf(t)
% respiration response function, as described in 
% 
% Chang, Catie, and Gary H. Glover. �Effects of Model-based Physiological 
% Noise Correction on Default Mode Network Anti-correlations and Correlations.� 
% NeuroImage 47, no. 4 (October 1, 2009): 1448�1459. doi:10.1016/j.neuroimage.2009.05.012.
%
% following:
% Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008. The respiration response
% function: the temporal dynamics of fMRI signal fluctuations related to changes in
% respiration. Neuroimage 40, 644�654
%
%
%   rrf = tapas_physio_rrf(t)
%
% IN
%   t       vector of timepoints
% OUT
%   rrf     respiration response function at sampled time points
% EXAMPLE
%
%   % just for visualization
%   t = 0:0.1:100;
%   rrf = tapas_physio_rrf(t);
%   figure;plot(t,rrf);
%
%   See also tapas_physio_crf

% Author: Lars Kasper
% Created: 2013-07-26
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

rrf = 0.6*t.^2.1.*exp(-t/1.6) - 0.0023*t.^3.54.*exp(-t/4.25);
