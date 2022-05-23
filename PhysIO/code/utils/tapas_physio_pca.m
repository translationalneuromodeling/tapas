function [eigenvariate, eigenvalues, eigenimage, vairance_explained, mean_across_voxels] = tapas_physio_pca( timeseries, verbose )
% Performes Principal Component Analysis (PCA).
% The functions uses the covariance matrix of input "timeseries"
% to allow obtaining PCs whatever the number of voxels.
%
% The code is adapted from "spm_run_voi" + "spm_regions"
%
% IN
%
%   timeserie          : [ nVolume , nVoxel ]
%
% OUT
%
%   eigenvariate       : if nVolume >  nVoxel : [ nVolume , nVoxel ] // if nVolume <= nVoxel : [ nVolume , nVolume ]
%   eigenvalues        :                        [ nVolume ,      1 ]
%   eigenimage         : if nVolume >  nVoxel : [ nVoxel  , nVoxel ] // if nVolume <= nVoxel : [ nVoxel  , nVolume ]
%   vairance_explained : in percent(%)          [ nVolume ,      1 ]
%   mean_across_voxels :                        [ nVolume ,      1 ]
%
%
% EXAMPLE
%
%   [eigenvariate, eigenvalues, eigenimage, vairance_explained, mean_across_voxels] = tapas_physio_pca( timeseries, verbose )
%
%
% NOTES
%
% An article that helped understanding PCA and SVD :
% https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
%
%
% See also tapas_physio_create_noise_rois_regressors spm_regions


%% Checks

% Each column is a voxel timeserie
[ nVolume , nVoxel ] = size(timeseries); % [ nVolume , nVoxel ]

% This can happen when your input mask covers more voxels then your fMRI
% volume, for exemple when you do not have a full brain acquisition
%
% This can also happens when you combine different toolboxs that have
% different methods to deal with the value outside a mask. For exemple,
% AFNI replace voxel value outise a maks by NaN. But SPM uses 0 outside the
% mask.
not_finite = ~isfinite(timeseries);
if any(not_finite(:))
    verbose = tapas_physio_log(...
        sprintf('[%s]: timeseries contains NaN or Inf, replacig it with 0\n', mfilename),...
        verbose, 0);
    timeseries(not_finite) = 0;
end


%% Center data

% First regressor : mean timeserie of the ROI
mean_across_voxels  = mean(timeseries,2); % [ nVolume ,      1 ]

% Center data : remove temporal mean, mandatory step to perform SVD
mean_across_volumes = mean(timeseries,1); % [       1 , nVoxel ]
timeseries_centered = timeseries - mean_across_volumes;


%% SVD

if nVolume > nVoxel
    [v,s,v] = svd(timeseries_centered' * timeseries_centered );
    u       =     timeseries_centered  * v/sqrt(s);
else
    [u,s,u] = svd(timeseries_centered  * timeseries_centered');
    v       =     timeseries_centered' * u/sqrt(s);
end

% Sign convention
d            = sign(sum(v));

% if nVolume >  nVoxel : [ nVolume , nVoxel  ]
% if nVolume <= nVoxel : [ nVolume , nVolume ]
eigenvariate = u.*d;

% if nVolume >  nVoxel : [ nVoxel , nVoxel  ]
% if nVolume <= nVoxel : [ nVoxel , nVolume ]
eigenimage   = v.*d;

% eigenvalues -> Variance explained
eigenvalues        = diag(s);                          %               [ nVolume , 1 ]
vairance_explained = 100*eigenvalues/sum(eigenvalues); % in percent(%) [ nVolume , 1 ]


end % function
