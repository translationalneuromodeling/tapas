function this = realign(this)
% Realigns all 3D images in 4D data to each other, then to the mean
% Uses SPM's realign: estimate+rewrite functionality
%
%   MrSeries = realign(MrSeries)
%
% This is a method of class MrSeries.
%
% IN
%   MrImage.data
%   MrImage.parameters.realign.quality
%
% OUT
%
% EXAMPLE
%   realign
%
%   See also MrSeries MrImage MrImage.realign

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-01
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


quality = this.parameters.realign.quality;

this.init_processing_step('realign');
this.data = this.data.realign('quality', quality);
this.finish_processing_step('realign', this.data);