%-----------------------------------------------------------------------
% Job saved on 01-Apr-2016 11:17:08 by cfg_util (rev $Rev: 6134 $)
% spm SPM - SPM12 (6225)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{2}.spm.stats.fmri_est.spmmat = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.space.volume.block_type = 'Slices';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.signal = 'Uninformative';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.ARP = 3;
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.noise.UGL = 1;
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.LogEv = 'Yes';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.anova.first = 'No';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.anova.second = 'Yes';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.gcon(1).name = 'main positive';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.gcon(1).convec = 1;
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.gcon(2).name = 'main negative';
matlabbatch{2}.spm.stats.fmri_est.method.Bayesian.gcon(2).convec = -1;
