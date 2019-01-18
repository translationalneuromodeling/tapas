function crf = tapas_physio_crf(t)
% cardiac response function, as described in 
% 
% Chang, Catie, and Gary H. Glover. �Effects of Model-based Physiological 
% Noise Correction on Default Mode Network Anti-correlations and Correlations.� 
% NeuroImage 47, no. 4 (October 1, 2009): 1448�1459. doi:10.1016/j.neuroimage.2009.05.012.
%
% following:
% Chang, C., Cunningham, J.P., Glover, G.H., 2009. Influence of heart rate 
% on the BOLD signal: the cardiac response function. Neuroimage 44, 857�869.%
%
%   crf = tapas_physio_crf(t)
%
% IN
%   t       vector of timepoints
% OUT
%   crf     cardiac response function at sampled time points
%
% EXAMPLE
%   % just for visualization
%   t = 0:0.1:100;
%   crf = tapas_physio_crf(t);
%   figure;plot(t,crf);
%
%   See also tapas_physio_rrf

% Author: Lars Kasper
% Created: 2013-07-26
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

crf = 0.6*t.^2.7.*exp(-t/1.6) - 16/sqrt(2*pi*9).*exp(-1/2.*(t-12).^2/9);
