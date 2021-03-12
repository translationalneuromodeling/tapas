function diffOddEven = compute_diff_odd_even(this, varargin)
% computes difference image between first and last volume of time series
% NOTE: short-cut for compute_stat_image('diff_last_first')
%
%   Y = MrImage()
%   diffOddEven = Y.compute_diff_odd_even('PropertyName', PropertyValue)
%
% This is a method of class MrImage.
%
% IN
%   'PropertyName'
%               'selectedVolumes'       [1,nVols] vector of selected
%                                       volumes for statistical calculation
% OUT
%   diffOddEven               MrImage holding difference image between 
%                               odd and even
%                               time series volume, characterizing "image
%                               noise" as in FBIRN paper (Friedman and
%                               Glover, JMRI 2006)
% EXAMPLE
%   Y = MrImage()
%   diffOddEven = Y.compute_diff_odd_even('selectedVolumes', [6:100])
%
%   See also MrImage compute_stat_image

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2020-09-23
% Copyright (C) 2020 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


diffOddEven = this.compute_stat_image('diff_last_first', varargin{:});
