function [coregisteredImage, affineCoregistrationGeometry, outputOtherImages] =...
    coregister_to(this, stationaryImage, varargin)
% Coregister this MrImage to another MrImage using SPM's coregister.
%
%   Y = MrImage()
%   cY = Y.coregister_to(stationaryImage)
%
% This is a method of class MrImage.
%
% IN
%   stationaryImage     cell {nImages,n} of MrImages or single MrImage that
%                       serves as stationary" or reference image to which
%                       this image is coregistered to
%
%  optional parameter name/value pairs:
%
%  transformation estimation and application
%       applyTransformation
%                   'geometry'      MrImageGeometry is updated,
%                                   MrImage.data remains untouched
%                   'data'          MrImage.data is resliced to new
%                                   geometry
%                                   NOTE: An existing
%                                   transformation in MrImageGeometry will
%                                   also be applied to MrImage, combined
%                                   with the calculated one for
%                                   coregistration
%                   'none'          transformation matrix is
%                                   computed, but not applied to geometry
%                                   of data of this image
%       trafoParameters             'translation', 'rigid', 'affine',
%                                   'rigidscaled' or
%                                   [1,1-12] vector of starting parameters
%                                   for transformation estimation number of
%                                   elements decides whether
%                                   translation only (1-3)
%                                   rigid (4-6)
%                                   rigid and scaling (7-9)
%                                   affine (10-12)
%                                   transformation is performed
%                                   default: 'rigid' (as in SPM)
% SPM input parameters:
%          separation           optimisation sampling steps (mm)
%                               default: [4 2]
%          objectiveFunction    cost function string:
%                               'mi'  - Mutual Information
%                               'nmi' - Normalised Mutual Information
%                               'ecc' - Entropy Correlation Coefficient
%                               'ncc' - Normalised Cross Correlation
%                               default: 'nmi'
%          tolerances           tolerances for accuracy of each param
%                               default: [0.02 0.02 0.02 0.001 0.001 0.001]
%          histSmoothingFwhm    smoothing to apply to 256x256 joint histogram
%                               default: [7 7]
%          otherImages          cell(nImages,1) of other images (either
%                               file names or MrImages) that should undergo
%                               the same trafo as this images due to coreg
%                               default: {}
%           doPlot              set to true for graphical output and PS file creation
%                               in SPM graphics window
%                               default: false
%
%   Parameters for high-dim application:
%       representationIndexArray:   either an MrImageObject or a selection
%                                   (e.g. {'echo', 1} which is then applied to
%                                   obtain one 4D image
%                                   default representationIndexArray: first
%                                   index of all extra (non-4D) dimensions
%       applicationIndexArray:      a selection which defines one or multiple
%                                   4D images on which the estimated parameters
%                                   are applied
%                                   default applicationIndexArray: all non-4D
%                                   dimensions
%       splitComplex                'ri' or 'mp'
%                                   If the data are complex numbers, real and imaginary or
%                                   magnitude and phase are realigned separately.
%                                   default: ri (real and imaginary)
%
% OUT
%
% EXAMPLE
%   coregister_to
%
%   See also MrImageSpm4D.coregister_to

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-28
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% SPM parameters
defaults.doPlot = false; % set to true for graphical output and PS file creation
defaults.otherImages = {};
defaults.objectiveFunction = 'nmi';
defaults.separation = [4 2 1];
defaults.tolerances = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
defaults.histSmoothingFwhm = [7 7];
defaults.trafoParameters = 'rigid';
% other coreg_to parameters of MrImageSpm4D
defaults.applyTransformation = 'data';
defaults.otherImages = {};
args = tapas_uniqc_propval(varargin, defaults);
% tapas_uniqc_strip_fields(args);

methodParameters = {args};

%% Distinguish cases

%% 1) this: 3D; stationary: 3D; other: nD
%  -> trivial case, directly mimicking SPM's coregister and passing to
%  MrImageSpm4D

thisNonSDims = this.dimInfo.get_non_singleton_dimensions;
thisIs3D = numel(thisNonSDims) == 3;
thisIsReal = isreal(this);
thisIsReal3D = thisIs3D && thisIsReal;

stationaryNonSDims = stationaryImage.dimInfo.get_non_singleton_dimensions;
stationaryIs3D = numel(stationaryNonSDims) == 3;
stationaryIsReal = isreal(stationaryImage);
stationaryIsReal3D = stationaryIs3D && stationaryIsReal;

if thisIsReal3D &&  stationaryIsReal3D % just do as SPM does!
    
    % for now, the stationary is 3D
    stationaryImage = stationaryImage.remove_dims();
    
    xyzDimLabels = {'x','y','z'};
    splitDimIndexArray = setdiff(1:this.dimInfo.nDims, ...
        this.dimInfo.get_dim_index(xyzDimLabels));
    splitDimLabels = this.dimInfo.dimLabels(splitDimIndexArray);
    
    [coregisteredImage, affineCoregistrationGeometry, outputOtherImages] = ...
        this.apply_spm_method_per_4d_split(...
        @(x, y) coregister_to(x, stationaryImage, y), ...
        'methodParameters', methodParameters, 'splitDimLabels', splitDimLabels);
    
end

end
%% 2) this: nD; stationary: 3D; other: cell(nSplits,nOtherImages) of 3D images
%% 2a) representation: 3D; application: nD
%  -> one 3D part represents this for the coregistration to stationary, and
%  the estimated coreg parameters are then applied to all images in the
%  application selections.
%% 2b) representation: nD; application: nD
%  -> each 3D part of "this" will be individually coregistered to stationary;
%     and kth coreg is applied to coresponding k-th cell of other images
%     Note: cell of nSplits of 3D images can be created by
%     other.split('splitDimLabels', {'t','echo'}) or similar
%% 3) this: 3D; stationary: cell(nStationary,1) of 3D images; other: cell (nStationary, nOtherImages)
%  -> "this" is coregistered to each of the stationaries (therefore 3+x D
%  output), and parameters of iStationary's coeregistration are applied to corresponding other images in cell {iStationary,:}
%% 4) this: nD; stationary: cell(nStationary,1) of 3D images; other cell(nStationary, nOtherImages)
%     Note: nStationary == nSplits of "this" here
%  -> Each 3D part of "this" is coregistered to the corresponding iSplit
%  (iStationary) entry of cell stationary, and the coreg parameters are
%  applied to all entries in cell {iStationary,:} of other images.
