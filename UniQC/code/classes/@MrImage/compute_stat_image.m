function statMrImage = compute_stat_image(this, statImageType, varargin)
% wrapper for computing different statistical images (taken over time series)
% while retaining image geometry information
%
%   Y = MrImage()
%   statMrImage = Y.compute_stat_image(statImageType, ...
%                                       'PropertyName', PropertyValue, ...)
%
% This is a method of class MrImage.
%
% IN
%   statImageType   'snr'       (default), ignoring voxels with sd < 1e-6
%                   'sd'        standard deviation,
%                   'mean'
%                   'coeffVar'  (coefficient of variance) = 1/snr;
%                               ignoring voxels with mean < 1e-6
%                   'diff_last_first' 
%                               difference image between last and first
%                               time series volume, characterizing drift
%                   'diff_odd_even' 
%                               difference image between odd and even
%                               time series volume, characterizing "image
%                               noise" as in FBIRN paper (Friedman and
%                               Glover, JMRI 2006)
%
%   'PropertyName'
%               'applicationDimension'  dimension along which statistical
%                                       calculation is performed
%                                       default: 't'
% OUT
%   statMrImage     output statistical image. See also MrImage
%
% EXAMPLE
%   Y = MrImage()
%   snr = Y.compute_stat_image('snr', 'applicationDimension', 't');
%
%   See also MrImage

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


defaults.applicationDimension = 't';

% fills in default arguments not given as input
args = tapas_uniqc_propval(varargin, defaults);

% strips input fields from args-structure,
% i.e. args.selectedVolumes => selectedVolumes
tapas_uniqc_strip_fields(args);

% get application index
applicationIndex = this.dimInfo.get_dim_index(applicationDimension);
% if applicationDimension is not found
if isempty(applicationIndex)
    if strcmp(applicationDimension, 't') % default has been used, alternative is last dim
        applicationIndex = this.ndims;
        applicationDimension = this.dimInfo.dimLabels(applicationIndex);
        applicationDimension = applicationDimension{1};
    else
        error('tapas:uniqc:MrImage:ApplicationDimensionDoesNotExist', ...
            'The specified application dimension %s does not exist.', ...
            applicationDimension);
    end
end

switch lower(statImageType)
    case 'mean'
        statMrImage = this.mean(applicationIndex);
    case 'sd'
        statMrImage = this.std(applicationIndex);
    case 'snr'
        tmpSd = threshold(this.std(applicationIndex), 1e-6); % to avoid divisions by zero
        statMrImage = this.mean(applicationIndex)./tmpSd;
        
    case {'coeffvar', 'coeff_var'}
        tmpMean = threshold(this.mean(applicationIndex), 1e-6);% to avoid divisions by zero
        statMrImage = this.std(applicationIndex)./tmpMean;
        
    case {'difflastfirst', 'diff_last_first'}
        statMrImage = this.select(applicationDimension, 1) - ...
            this.select(applicationDimension, ...
            this.dimInfo.(applicationDimension).nSamples(end));
        
    case {'diffoddeven', 'diff_odd_even'}
        nSamples = this.dimInfo.(applicationDimension).nSamples(end);
        statMrImage = this.select(applicationDimension, 1:2:nSamples) - ...
            this.select(applicationDimension, ...
            2:2:nSamples);
end

statMrImage.name = sprintf('%s (%s)', statImageType, this.name);

end
