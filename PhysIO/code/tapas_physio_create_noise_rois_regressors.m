function [R_noise_rois, noise_rois, verbose] = tapas_physio_create_noise_rois_regressors(...
    noise_rois, verbose)
% Compute physiological regressors by extracting principal components of
% the preprocessed fMRI time series from anatomically defined noise ROIS (e.g. CSF)
%
% [R_noise_rois, verbose] = tapas_physio_create_noise_rois_regressors(...
%     noise_rois, verbose)
%
% NOTE: The mean of all time series in each ROI will also be added as a regressor
% automatically
%
% Approach similar to the one described as aCompCor:
% Behzadi, Y., Restom, K., Liau, J., Liu, T.T., 2007. 
% A component based noise correction method (CompCor) for BOLD and 
% perfusion based fMRI. NeuroImage 37, 90?101. 
% doi:10.1016/j.neuroimage.2007.04.042
%
% IN
%   physio.model.noise_rois
% OUT
%
% EXAMPLE
%   tapas_physio_create_noise_rois_regressors
%
%   See also spm_ov_roi
%
% Author: Lars Kasper
% Created: 2015-07-22
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_create_noise_rois_regressors.m 810 2015-08-12 00:50:31Z kasperla $

% TODO: visualization of Rois, components/mean and spatial loads in ROIs

tapas_physio_strip_fields(noise_rois);

if isempty(fmri_files) || isempty(roi_files)
    R_noise_rois = [];
    return
end

if ~iscell(fmri_files)
    fmri_files = cellstr(fmri_files);
end

if ~iscell(roi_files)
    roi_files = cellstr(roi_files);
end

nRois = numel(roi_files);

if numel(thresholds) == 1
    thresholds = repmat(thresholds, 1, nRois);
end

if numel(n_voxel_crop) == 1
    n_voxel_crop = repmat(n_voxel_crop, 1, nRois);
end

if numel(n_components) == 1
    n_components = repmat(n_components, 1, nRois);
end

% TODO: what if different geometry of mask and fmri data?
%       or several fmri files given?
Vimg = spm_vol(fmri_files{1});
Yimg = spm_read_vols(Vimg);

nVolumes = size(Yimg, 4);
Yimg = reshape(Yimg, [], nVolumes);

R_noise_rois = [];
for r = 1:nRois
    
    Vroi = spm_vol(roi_files{r});
    roi = spm_read_vols(Vroi);
    roi(roi < thresholds(r)) = 0;
    roi(roi >= thresholds(r)) = 1;
    
    % crop pixel, if desired
    if n_voxel_crop(r)
        nSlices = size(roi,3);
        for s = 1:nSlices
            roi(:,:,s) = imerode(roi(:,:,s), strel('disk', n_voxel_crop(r)));
        end
    end
       
    Yroi = Yimg(roi(:)==1, :);
    
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
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(Yroi);
    
    % components either via number or threshold of variance explained
    if n_components(r) >= 1
        nComponents = n_components(r); 
    else
        nComponents = find(cumsum(EXPLAINED)/100 > n_components(r), 1, 'first');
    end
    
    % save to return
    noise_rois.n_components(r) = nComponents + 1; % + 1 for mean
    
    % Take mean and some components into noise regressor
    R = MU';
    R = [R, COEFF(:,1:nComponents)];
    
    nRegressors = size(R,2);
    
    % z-transform
    stringLegend = cell(nRegressors,1);
    for c = 1:nRegressors
        R(:,c) = (R(:,c) - mean(R(:,c)))/std(R(:,c));
        
        if c > 1
            stringLegend{c} = ...
                sprintf('Principal component %d (%7.4f %% variance explained)', ...
                c-1, EXPLAINED(c-1));
        else
            stringLegend{c} = 'Mean time series of all voxels';
        end
    end
    
    if verbose.level >=2
        stringFig = sprintf('Noise_rois: Extracted principal components for ROI %d', r);
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params(); 
        set(gcf, 'Name', stringFig);
        plot(R);
        legend(stringLegend);
        title(stringFig);
        xlabel('scans');
        ylabel('PCs, mean-centred and std-scaled');
    end
    
    % write away extracted PC-loads
    [tmp,fnRoi] = fileparts(Vroi(1).fname);
    fpFmri = fileparts(Vimg(1).fname);
    for c = 1:nComponents
        Vpc= Vroi;
        Vpc.fname = fullfile(fpFmri, sprintf('pc%02d_scores_%s.nii',c, fnRoi));
        pcScores = zeros(Vpc.dim);
        pcScores(roi(:)==1) = SCORE(:, c);
        spm_write_vol(Vpc, pcScores);
    end
    R_noise_rois = [R_noise_rois, R];
end