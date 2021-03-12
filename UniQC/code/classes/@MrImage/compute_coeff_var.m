function coeffVar = compute_coeff_var(this, varargin)
% computes standard deviation image (coeff_var) over 4th dimension of MrImage
% NOTE: short-cut for compute_stat_image('coeff_var')
%
%   Y = MrImage()
%   coeffVar = Y.compute_coeff_var('PropertyName', PropertyValue)
%
% This is a method of class MrImage.
%
% IN
%   'PropertyName'
%               'selectedVolumes'       [1,nVols] vector of selected
%                                       volumes for statistical calculation
% OUT
%   coeffVar         MrImage holding voxel-wise coefficient of variation
%                     image (coeff_var), i.e. 1./snr 
%                     (with thresholding to avoid Inf-values)
%
% EXAMPLE
%   Y = MrImage()
%   coeffVar = Y.compute_coeff_var('selectedVolumes', [6:100])
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


coeffVar = this.compute_stat_image('coeff_var', varargin{:});
