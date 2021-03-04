function outputImage = kfilter(this, filterType, applicationDimensions, varargin)
% Filters by multiplication of a specified window function in k-space
%
%   Y = MrDataNd()
%   outputImage = Y.kfilter(filterType, applicationDimensions)
%
% This is a method of class MrDataNd.
%
% IN
%   filterType  string of filter to be applied, possible values are
%               'hamming' (default)
%               'hanning'
%               'raised_cosine'
%   applicationDimensions
%               '2D' or '3D'
%                   default: '2D' for single slices, '3D' otherwise
%               '2D' performs the filter slice-wise with the same filter,
%               '3D' performs the filter for a 3D symmetric version of the
%                    filter
%   varargin
%               extra filter parameters, depending on the chosen filter
%               'raised_cosine'
%               kfilter('raised_cosine', '2D' or '3D', 'fractionFOV', 0.5, ...
%               'beta', 0.5)
%                   fractionFOV  - fraction of FOV (1-dim!) where filter
%                                  reaches half Maximum
%                                  default: 0.5
%                   beta         - roll-off factor between 0 and 1 for the 
%                                  raised-cosine window 
%                                  (0 giving a box-car function, 
%                                  and 1 a cosine without plateau)
%                                  default: 0.5
%   doPlotFilter    true of false (default)
%                   if true, an extra plot is generated, showing the 
%                   filter response in k-space alongside central x- and
%                   y-line profiles of the image
%
% OUT
%
% EXAMPLE
%   kfilter
%
%   See also MrDataNd

% Author:   Lars Kasper, based on code by Johanna Vannesjo for 1D filtering
% Created:  2018-11-07
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.fractionFOV = 0.5;
defaults.beta = 0.5;
defaults.doPlotFilter = false;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

is3D = ndims(this) >= 2;

if nargin < 2
    filterType = 'hamming';
end

if nargin < 3
    if is3D
        applicationDimensions = '3D';
    else
        applicationDimensions = '2D';
    end
end

switch filterType
    case 'raised_cosine'
        % from J. Vannesjo, utils/general/raised_cosine.m Recon5-6, IBT
        funFilter = @(x) tapas_uniqc_raised_cosine((1:x) - floor(x/2), ...
            1/(fractionFOV*x), beta);
    otherwise
        funFilter = str2func(filterType);
end

dimInfoFilter = this.dimInfo.copyobj();

% column vector * row vector = matrix coordinate-wise product
filterMatrix = reshape(funFilter(this.dimInfo.nSamples(1)), [],1)*...
    reshape(funFilter(this.dimInfo.nSamples(2)), 1, []);

if doPlotFilter
    filterProfile = reshape(funFilter(this.dimInfo.nSamples(1)),[],1);
    figure('Name', 'k-filter profile');
    plot(filterProfile);
    xlim([1,this.dimInfo.nSamples(1)]);
    hold all;
    kDataProfile = this.image2k.abs.data(:,round(this.dimInfo.nSamples(2)/2), ...
        round(this.dimInfo.nSamples(3)/2));
    plot(kDataProfile/max(kDataProfile));
    plot(kDataProfile/max(kDataProfile).*filterProfile);
    legend('kfilter', 'unfiltered kx-profile of central slice', 'kfiltered kx-profile');
end

% replicate same filter for all slices
if is3D
    filterMatrix = repmat(filterMatrix, 1, 1, this.dimInfo.nSamples(3));
    dimInfoFilter.remove_dims(4:dimInfoFilter.nDims);
    
    switch applicationDimensions
        case '3D'
            % create the filter in 3rd dimension by replicating in other 2
            % dims and multiplying with slice-replicated 2D-filter
            filterMatrixThirdDim = reshape(funFilter(this.dimInfo.nSamples(3)), 1, 1, []);
            filterMatrix = filterMatrix.*repmat(filterMatrixThirdDim,...
                dimInfoFilter.nSamples(1), dimInfoFilter.nSamples(2), 1);
        case '2D'
            % everything fine, we just replicated for all slices
    end
    
else
    dimInfoFilter.remove_dims(3:dimInfoFilter.nDims);
end


filterImage = MrImage(filterMatrix, 'dimInfo', dimInfoFilter);
outputImage = k2image(image2k(this, applicationDimensions).*filterImage, ...
    applicationDimensions);