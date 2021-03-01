function this = t_filter(this)
% high-pass filters temporally (4th dimension of the image) as SPM
%
%   MrSeries = t_filter(MrSeries)
%
% This is a method of class MrSeries.
%
% IN
%   parameters.TR_s
%   parameters.t_filter.cutoffSeconds
%
% OUT
%
% EXAMPLE
%   t_filter
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-02
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


this.init_processing_step('t_filter');
this.data.t_filter(this.parameters.t_filter.cutoffSeconds);
this.finish_processing_step('t_filter', this.data);