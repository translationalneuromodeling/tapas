%-----------------------------------------------------------------------
% Job saved on 06-Jan-2015 09:56:59 by cfg_util (rev $Rev$)
% spm SPM - SPM12 (6225)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.tools.physio.save_dir = {'physio_out'};
matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Philips';
matlabbatch{1}.spm.tools.physio.log_files.cardiac = {'SCANPHYSLOG.log'};
matlabbatch{1}.spm.tools.physio.log_files.respiration = {'SCANPHYSLOG.log'};
matlabbatch{1}.spm.tools.physio.log_files.scan_timing = {''};
matlabbatch{1}.spm.tools.physio.log_files.sampling_interval = [];
matlabbatch{1}.spm.tools.physio.log_files.relative_start_acquisition = 0;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nslices = 36;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.TR = 2;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Ndummies = 3;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nscans = 230;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.onset_slice = 18;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.c = 3;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.r = 4;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.cr = 1;
matlabbatch{1}.spm.tools.physio.model.rvt.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.hrv.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.noise_rois.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.orthogonalise = 'none';
matlabbatch{1}.spm.tools.physio.model.movement.yes.file_realignment_parameters = {'rp_fMRI.txt'};
matlabbatch{1}.spm.tools.physio.model.movement.yes.order = 6;
matlabbatch{1}.spm.tools.physio.model.movement.yes.censoring_method = 'MAXVAL';
matlabbatch{1}.spm.tools.physio.model.movement.yes.censoring_threshold = [3 Inf];
matlabbatch{1}.spm.tools.physio.model.output_multiple_regressors = 'multiple_regressors.txt';
matlabbatch{1}.spm.tools.physio.scan_timing.sync.gradient_log.grad_direction = 'y';
matlabbatch{1}.spm.tools.physio.scan_timing.sync.gradient_log.zero = 1500;
matlabbatch{1}.spm.tools.physio.scan_timing.sync.gradient_log.slice = 2200;
matlabbatch{1}.spm.tools.physio.scan_timing.sync.gradient_log.vol = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sync.gradient_log.vol_spacing = 0.09;
matlabbatch{1}.spm.tools.physio.preproc.cardiac.modality = 'ECG';
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
matlabbatch{1}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
matlabbatch{1}.spm.tools.physio.verbose.level = 2;
matlabbatch{1}.spm.tools.physio.verbose.fig_output_file = 'PhysIO_output.fig';
matlabbatch{1}.spm.tools.physio.verbose.use_tabs = false;
