function diffLastFirst = compute_diff_last_first(this, varargin)
% computes difference image between first and last volume of time series
% NOTE: short-cut for compute_stat_image('diff_last_first')
%
%   Y = MrImage()
%   diffLastFirst = Y.compute_diff_last_first('PropertyName', PropertyValue)
%
% This is a method of class MrImage.
%
% IN
%   'PropertyName'
%               'selectedVolumes'       [1,nVols] vector of selected
%                                       volumes for statistical calculation
% OUT
%   diffLastFirst         MrImage holding difference image between
%                         last and first time series volume, 
%                         characterizing drift
%
% EXAMPLE
%   Y = MrImage()
%   diffLastFirst = Y.compute_diff_last_first('selectedVolumes', [6:100])
%
%   See also MrImage compute_stat_image

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-06
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


diffLastFirst = this.compute_stat_image('diff_last_first', varargin{:});
