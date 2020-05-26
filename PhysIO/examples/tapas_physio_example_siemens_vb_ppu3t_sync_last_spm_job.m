%-----------------------------------------------------------------------
% Job saved on 28-Jan-2020 11:28:55 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7487)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.tools.physio.save_dir = {'physio_out'};
matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Siemens';
matlabbatch{1}.spm.tools.physio.log_files.cardiac = {'20191023_V02.puls'};
matlabbatch{1}.spm.tools.physio.log_files.respiration = {'20191023_V02.resp'};
matlabbatch{1}.spm.tools.physio.log_files.scan_timing = {'MR.27842.1611'};
matlabbatch{1}.spm.tools.physio.log_files.sampling_interval = 0.0025;
matlabbatch{1}.spm.tools.physio.log_files.relative_start_acquisition = [];
matlabbatch{1}.spm.tools.physio.log_files.align_scan = 'last';
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nslices = 24;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.TR = 1.5;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Ndummies = 0;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nscans = 382;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.onset_slice = 5;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = 0.0625;
matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
matlabbatch{1}.spm.tools.physio.scan_timing.sync.scan_timing_log = struct([]);
matlabbatch{1}.spm.tools.physio.preproc.cardiac.modality = 'PPU';
matlabbatch{1}.spm.tools.physio.preproc.cardiac.filter.no = struct([]);
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.max_heart_rate_bpm = 120;
matlabbatch{1}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
matlabbatch{1}.spm.tools.physio.model.output_multiple_regressors = 'multiple_regressors.txt';
matlabbatch{1}.spm.tools.physio.model.output_physio = 'physio.mat';
matlabbatch{1}.spm.tools.physio.model.orthogonalise = 'none';
matlabbatch{1}.spm.tools.physio.model.censor_unreliable_recording_intervals = false;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.c = 3;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.r = 4;
matlabbatch{1}.spm.tools.physio.model.retroicor.yes.order.cr = 1;
matlabbatch{1}.spm.tools.physio.model.rvt.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.hrv.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.noise_rois.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.movement.no = struct([]);
matlabbatch{1}.spm.tools.physio.model.other.no = struct([]);
matlabbatch{1}.spm.tools.physio.verbose.level = 2;
matlabbatch{1}.spm.tools.physio.verbose.fig_output_file = '';
matlabbatch{1}.spm.tools.physio.verbose.use_tabs = false;
