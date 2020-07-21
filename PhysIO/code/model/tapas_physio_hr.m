function hr = tapas_physio_hr(cpulse, t)
% computes average heart rate at predefined times from time series of cardiac pulse events
%
%    hr = tapas_physio_hr(cpulse, t)
%
% a sliding window of 6 seconds around the time points is used to
% determine heart rate variability, as suggested in 
%
% Chang, Catie, John P. Cunningham, and Gary H. Glover. 
% Influence of Heart Rate on the BOLD Signal: The Cardiac Response Function.
% NeuroImage 44, no. 3 (February 1, 2009): 857-869. 
% doi:10.1016/j.neuroimage.2008.09.029.
%
% IN
%   cpulse  onset times (seconds) of heartbeat pulses (R-wave peak)
%   t       vector of time points (seconds) heart rate should be calculated
% OUT
%   hr      vector of average heart rate (in beats per minute) at specified time points
%
% EXAMPLE
%   hr = tapas_physio_hr(physio_out.ons_secs.cpulse,physio_out.ons_secs.svolpulse)
%
%   See also tapas_physio_create_hrv_regressor

% Author: Lars Kasper
% Created: 2013-07-26
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


n = length(t);

sws = 6; % sliding window width

hr = zeros(size(t));
for i = 1:n
   iL = find(cpulse > t(i) - sws/2,1,'first') - 1;   % -1 to compute heart rate at all beats within the sliding window
   iR = find(cpulse < t(i) + sws/2, 1, 'last') + 1;
   iL = max(1, iL);
   iR = min(iR, length(cpulse));
   if ~isempty(iL) && ~isempty(iR)
    hr(i) = 1./mean(diff(cpulse(iL:iR)))*60;
   end
end
