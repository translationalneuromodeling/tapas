function this = smooth(this)
% Smoothes all images in the time series
%
%   MrSeries = smooth(MrSeries)
%
% This is a method of class MrSeries.
%
% IN
%   parameters.smooth.fwhmMillimeter
%
% OUT
%
% EXAMPLE
%   smooth
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


fwhm = this.parameters.smooth.fwhmMillimeters;

this.init_processing_step('smooth');
this.data = this.data.smooth('fwhm', fwhm);
this.finish_processing_step('smooth', this.data);