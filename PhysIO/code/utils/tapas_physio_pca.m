function [principal_component, mean_across_voxels, eigen_values, vairance_explained, load] = tapas_physio_pca( timeseries, verbose )
% Performes Principal Component Analysis (PCA).
% The functions uses the covariance matrix of input "timeseries"
% to allow obtaining PCs whatever the number of voxels.
%
% The code is adapted from "spm_run_voi" + "spm_regions"
%
% IN
%
%   timeserie         : [ nVolume , nVoxel ]
%
% OUT
%
%   principal_component : if nVolume >  nVoxel : [ nVolume , nVoxel ] // if nVolume <= nVoxel : [ nVolume , nVolume ]
%   mean_across_voxels  :                        [ nVolume ,      1 ]
%   eigen_values        :                        [ nVolume ,      1 ]
%   vairance_explained  : in percent(%)          [ nVolume ,      1 ]
%   load                : if nVolume >  nVoxel : [ nVoxel  , nVoxel ] // if nVolume <= nVoxel : [ nVoxel  , nVolume ]
%
%
% EXAMPLE
%
%   [principal_component, mean_across_voxels, eigen_values, vairance_explained, load] = tapas_physio_pca( timeseries, verbose )
%
%
% See also tapas_physio_create_noise_rois_regressors


%% Checks

% Each column is a voxel timeserie
[ nVolume , nVoxel ] = size(timeseries); % [ nVolume , nVoxel ]

% This can happens when you combine different toolboxs that have different
% methods to deal with the value outside a mask. For exemple, AFNI replace
% voxel value outise a maks by NaN. But SPM uses 0 outside the mask.
not_finite = ~isfinite(timeseries);
if any(not_finite(:))
    verbose = tapas_physio_log(...
        sprintf('[%s]: timeseries contains NaN or Inf, replacig it with 0 : %s \n', mfilename),...
        verbose, 1);
    timeseries(not_finite) = 0;
end


%% Center data

% First regressor : mean timeserie of the ROI
mean_across_voxels = mean(timeseries,2); % [ nVolume , 1 ]

% Center data : remove temporal mean, mandatory step to perform SVD
mean_across_volumes = mean(timeseries,1); % [ 1 , nVoxel ]
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
d                   = sign(sum(v));

% if nVolume >  nVoxel : [ nVolume , nVoxel  ]
% if nVolume <= nVoxel : [ nVolume , nVolume ]
principal_component = u.*d;

% if nVolume >  nVoxel : [ nVoxel , nVoxel  ]
% if nVolume <= nVoxel : [ nVoxel , nVolume ]
load                = v.*d;


%% Diagnostics

% Singular values -> Eigen values
singular_values = diag(s);
eigen_values    = singular_values.^2/(nVoxel-1); % [ nVolume , 1 ]

% Eigen_values -> Variance explained
vairance_explained = 100*eigen_values/sum(eigen_values); % in percent(%) [ nVolume , 1 ]


end % function
