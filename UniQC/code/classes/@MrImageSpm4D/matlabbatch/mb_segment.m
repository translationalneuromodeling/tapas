%-----------------------------------------------------------------------
% Job saved on 14-Jul-2014 18:18:40 by cfg_util (rev $Rev$)
% spm SPM - SPM12b (5672)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.spatial.preproc.channel.vols = {'/Users/kasperla/Documents/code/matlab/fmri_svn/development/analysis/examples/model_based_fmri_3T/meanrest.nii,1'};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
% two entries (each 0 or 1): write bias-field and/or bias-corrected image 
matlabbatch{1}.spm.spatial.preproc.channel.write = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'/Users/kasperla/Documents/code/matlab/spm12b/tpm/TPM.nii,1'};
matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;

% two entries (each 0 or 1): write out data in [native space, Dartel-importable] form
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [0 0]; % 

% two entries: [modulated, unmodulated] modulated means correcting for 
% volume differences in tissues between subject and template
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0]; 
matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'/Users/kasperla/Documents/code/matlab/spm12b/tpm/TPM.nii,2'};

matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'/Users/kasperla/Documents/code/matlab/spm12b/tpm/TPM.nii,3'};
matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'/Users/kasperla/Documents/code/matlab/spm12b/tpm/TPM.nii,4'};
matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'/Users/kasperla/Documents/code/matlab/spm12b/tpm/TPM.nii,5'};
matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'/Users/kasperla/Documents/code/matlab/spm12b/tpm/TPM.nii,6'};
matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;

% two entires (0 or 1 each): write [inverse deformation field, 
% forward deformation field, 
% where: forward means mapping from subject to standard (mni) space
matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
