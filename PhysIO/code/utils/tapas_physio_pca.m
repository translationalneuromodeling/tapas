function [COEFF, SCORE, LATENT, EXPLAINED, MU] = tapas_physio_pca( timeseries, method )
% Performes the Principal Component Analysis (PCA)
%
%   [COEFF, SCORE, LATENT, EXPLAINED, MU] = tapas_physio_pca( timeseries )
%
% IN
%   timeserie                 (nVoxels x nVolumes)
%   method                    'svd' or 'stats-pca'
%
% OUT
%   COEFF = [nVolumes, nPCs]  principal components (PCs) ordered by variance
%                             explained
%   SCORE = [nVoxel, nPCs]    loads of each component in each voxel, i.e.
%                             specific contribution of each component in
%                             a voxel's variance
%   LATENT = [nPCs, 1]        eigenvalues of data covariance matrix,
%                             stating how much variance was explained by
%                             each PC overall
%   TSQUARED = [nVoxels,1]    Hotelling's T-Squared test whether PC
%                             explained significant variance in a voxel
%   EXPLAINED = [nPCs, 1]     relative amount of variance explained (in
%                             percent) by each component
%   MU = [1, nVolumes]        mean of all time series
%
% EXAMPLE
%   [COEFF, SCORE, LATENT, EXPLAINED, MU] = tapas_physio_pca( Yroi )
%
% See also tapas_physio_create_noise_rois_regressors

[nVoxels,nVolumes] = size(timeseries);

if nVoxels <= nVolumes
    error([mfilename ':NotEnoughVoxels'], 'nVoxels <= nVolumes')
end

if nargin < 2
    method = 'svd';
end

switch lower(method)
    
    case 'svd'
        
        % First regressor : mean timeserie of the ROI
        MU = mean(timeseries); % [1, nVolumes]
        
        % Center data : remove mean, mandatory step to perform SVD
        timeseries = timeseries - MU;
        
        % Perform Singular Value Ddecomposition
        [u,s,v] = svd(timeseries,0);
        
        % Singular values -> Eigen values
        singular_values = diag(s);
        eigen_values    = singular_values.^2/(nVoxels-1);
        LATENT          = eigen_values; % [nPCs, 1]
        
        % Eigen_values -> Variance explained
        vairance_explained = 100*eigen_values/sum(eigen_values); % in percent (%)
        EXPLAINED          = vairance_explained;                 % [nPCs, 1]
        
        % Sign convention : the max(abs(PCs)) is positive
        [~,maxabs_idx] = max(abs(v));
        [m,n]          = size(v);
        idx            = 0:m:(n-1)*m;
        val            = v(maxabs_idx + idx);
        sgn            = sign(val);
        v              = v .* sgn;
        u              = u .* sgn;
        
        COEFF = v;                     % [nVolumes, nPCs]
        SCORE = u .* singular_values'; % [nVoxel  , nPCs]
        
    case 'stats-pca'
        
        [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(timeseries);
        
    otherwise
        
        error('unrecognized method : ''svd'' of ''stats-pca'' are accepted')
        
end

end % function
