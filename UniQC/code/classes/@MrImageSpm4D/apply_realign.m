function realignedImage = apply_realign(this, rp, varargin)
% applies realignment parameters from other estimation to this 4D Image
%
%   Y = MrImageSpm4D()
%   rY = Y.apply_realign(rp, 'interpolation',7, 'masking', 1, 'wrapping', [0 0 0])
%
% This is a method of class MrImageSpm4D.
%
% IN
%   rp  [nVolumes,6] realignment parameters, as output by SPM 
%                    (e.g., in rp_*.txt)
%                    in mm and rad: [dx,dy,dz,pitch,roll,yaw]
%                                           (i.e., phi_x,phi_y,phi_z)
%   optional: parameterName/value pairs of
%   most SPM realign est/reslice parameters, enforcing congruency between 
%   est/reslice and ignoring file naming options:
%
%   interpolation   degree of b-spline interpolation for estimation and reslicing
%                   default: 7
%   wrapping        fold-over direction (phase encode)
%                   default: [0 0 0] % none
%   masking         mask incomplete timeseries?
%                   default: true
% 
% OUT
%
% EXAMPLE
%   apply_realign
%
%   See also MrImageSpm4D spm_run_coreg spm_matrix

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-25
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.interpolation = 7;     % degree of b-spline interpolation for estimation and reslicing
defaults.wrapping = [0 0 0];    % fold-over direction (phase encode)
defaults.masking = 1;           % mask incomplete timeseries?
% the following ones are used for consistency when retrieving
% realign-matlabbatch, but have no effect for reslicing here
defaults.quality = 0.9;         % 0..1, estimation quality, share of voxels included in estimation
defaults.separation = 4;        % separation distance (mm) between evaluated image points in estimation
defaults.smoothingFwhm = 5;     % FWHM (mm) of Gaussian smoothing kernel used for estimation
defaults.realignToMean = 1;     % boolean; if true, 2-pass procedure, registering to mean
defaults.weighting = '';        % weighting image for estimation

spmParameters = tapas_uniqc_propval(varargin, defaults);


%% save image file for processing as nii in SPM
realignedImage = this.copyobj;

realignedImage.save('fileName', realignedImage.get_filename('prefix', 'raw'));

[pathRaw, fileRaw, ext] = fileparts(realignedImage.get_filename('prefix', 'raw'));
fileRaw = [fileRaw ext];
PO = cellstr(spm_select('ExtFPList', pathRaw, ['^' fileRaw], Inf));


%% loop over volumes, Adapting image headers
% applying realignment as relative trafo to existing
% voxel/world mapping, only header (.mat) changed
% 1) loop code analogous to spm_run_coreg, around line 30 (eoptions)
% 2) BUT: application of matrices as inversion of spm_realign>save_parameters
%    (l. 533), since rps as saved as 'spm_imatrix(V(j).mat/V(1).mat)'

% 12 parameters of affine mapping, see spm_matrix for their order
x = zeros(1,12);
x(7:9) = 1;

for j = 1:numel(PO)
    x(1:6)  = rp(j,:);
    M  = spm_matrix(x);
    MM = spm_get_space(PO{j});
    spm_get_space(PO{j}, M*MM);
end

matlabbatch = realignedImage.get_matlabbatch('realign', spmParameters);
job = matlabbatch{1}.spm.spatial.realign.estwrite;


%% Reslicing has to happen here
% has to be before finish, since .mat 4D information not saved by our ...
% objects on this.load('*.nii')

% from spm_run_realign, around line 36, "if isfield(job,'roptions')" etc.
P            = char(PO);
flags.mask   = job.roptions.mask;
flags.interp = job.roptions.interp;
flags.which  = [2 0]; % don't write out mean, but all images
flags.wrap   = job.roptions.wrap;
flags.prefix = job.roptions.prefix;

spm_reslice(P, flags); 
realignedImage.finish_processing_step('apply_realign');
