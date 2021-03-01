function PCs = pca(this, cutoff, pcDimension)
% Computes principal component analysis (PCA) of image time series
%
%   Y = MrImage()
%   PCs = pca(this, cutoff, pcDimension)
%
% This is a method of class MrImage.
%
% Spatial PCA (default):
% The principal component analysis report representative spatial
% distributions ("principal components") of fluctuation patterns that explain most variance in the
% data (over all time points).
% The corresponding temporal evolution of weights (or scores or
% projections) describes how each of these spatial patterns fluctuates over
% time.
%
% Temporal PCA (?)
% The principal component analysis reports the representative time series
% ("principal components") that explain most of the variance in the data
% (pooled over all voxels).
% The corresponding spatial maps of weights (or scores or projections) give
% the relative contribution of this particular time series to the variance
% in each voxel
%
% IN
%   cutoff      value determining number of principal components extracted
%               0 < cutoff < 1      interpreted as relative amount of
%                                   variance explained;
%                                   nPCs will be determined as the number
%                                   of components that explain at least
%                                   cutoff*100 % of the variance in the
%                                   data
%               cutoff = 1,2,3...   interpreted as number of PCs extracted
%   pcDimension 'spatial' (default) or 'temporal'
%               determines which dimension i.e. spatial image or time
%               (volumes) will be principal component, and consequently,
%               which other one will be the projection dimension
%               Technically, the non-PC dimension is the one considered to
%               "generate the variance", in that the covariance matrix
%               entries would relate PC vector components, co-varying over
%               the non-PC dimension,
%               e.g.    spatial PCA: How do two voxels co-vary over time
%                       temporal PCA: How do two time-points co-vary over
%                       space
%               Typically, the 1st projection of the temporal PCA looks
%               like the mean, and for the other components, it usually
%               holds:
%               PC_spatial(n) approximately equal to Proj_temporal(n+1)
% OUT
%   PCs         cell(nPCs,1) of MrImages
%               Each element is a 4D image, the nth volume is computed as
%                   PCs{k}_n = PC{k}*Projection_n,
%               i.e. for
%                   spatial PCA: principal component * nth time point pf
%                   projection
%                   temporal PCA: nth element of principal component *
%                   projection vector of all voxels
%
% EXAMPLE
%   pca
%
%   See also MrImage

% Author:   Lars Kasper
% Created:  2015-08-13
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% if number of components specified explicity, no additional variance
% threshold is needed

if nargin < 2
    cutoff = 1;
end

isSpatialPca = nargin < 3 || strcmpi(pcDimension, 'spatial');

hasVarianceThreshold = cutoff<1;

if hasVarianceThreshold
    nComponents = 1;
else
    nComponents = cutoff;
    cutoff = 0;
end

% created reshaped data matrix for PCA

% data matrix X: [nVoxels, nVolumes] for 4D, [nVoxel2D, nSlices] for 3D...
applicationDimension = find(this.geometry.nVoxels>1, 1, 'last');
X = reshape(this.data, [], this.geometry.nVoxels(applicationDimension));

% [nVolumes, nVoxels] for spatial PCA...
if isSpatialPca
    X = X';
end

% remove invalid data for PCA
X(isinf(X)) = 0;
X(isnan(X)) = 0;

% iteratively increase number of components, if variance threshold given,
% otherwise compute once with number of components specified
doPca = 1;
while doPca
    
    
    % Explanation for temporal PCA
    % COEFF = [nVolumes, nPCs]  principal components (PCs) ordered by variance
    %                           explained
    % SCORE = [nVoxel, nPCs]    loads of each component in each voxel, i.e.
    %                           specific contribution of each component in
    %                           a voxel's variance
    % LATENT = [nPCs, 1]        eigenvalues of data covariance matrix,
    %                           stating how much variance was explained by
    %                           each PC overall
    % TSQUARED = [nVoxels,1]    Hotelling's T-Squared test whether PC
    %                           explained significant variance in a voxel
    % EXPLAINED = [nPCs, 1]     relative amount of variance explained (in
    %                           percent) by each component
    % MU = [1, nVolumes]        mean of all time series
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = ...
        pca(X, 'NumComponents', nComponents);
    
    explainedVariance   = sum(EXPLAINED(1:nComponents))/100;
    doPca               = hasVarianceThreshold && ...
        (explainedVariance < cutoff);
    
    if doPca
        nComponents = nComponents + 1;
        
        % somehow, pca also gives out EXPLAINED variance also for more than one
        % component, jump to 1st achieved cutoff-threshold
        nComponentsTemp = find(cumsum(EXPLAINED) >= cutoff*100, 1, 'first');
        if ~isempty(nComponentsTemp)
            nComponents = nComponentsTemp;
        end
    end
end

% create 4D PCs i.e. 3D Pcs which are co-varied along the volume
% dimension with the computed projections

PCs = cell(nComponents,1);
for c = 1:nComponents
    PCs{c} = this.copyobj;
    PCs{c}.rois = {};
    PCs{c}.name = sprintf(['PC %d (explained variance: %5.2f %%, ', ...
        'cumulative %5.2f %%)) - %s'], ...
        c, EXPLAINED(c), sum(EXPLAINED(1:c)), this.name);
    
    % multiply the PC with each weight entry to generate whole time series
    Y = kron(SCORE(:,c)', COEFF(:,c));
    
    if ~isSpatialPca
        Y = Y';
    end
    
    PCs{c}.data = reshape(Y, this.geometry.nVoxels);
end