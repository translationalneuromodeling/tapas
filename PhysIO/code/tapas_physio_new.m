function physio = tapas_physio_new(default_scheme, physio_in)
% Creates complete PhysIO structure fed into tapas_physio_main_create_regressors
%
%    physio = tapas_physio_new(default_scheme, physio_in)
%
% IN
%   default_scheme  - if set, default values for structure entries are set
%                       according to the application
%                       different templates are predefined, e.g.
%                       'empty'   - all strings are set to '', all
%                                     numbers to [] (default)
%                       'Philips':  good initial values for scans acquired
%                                   on a Philips 3T system
%                               model: RETROICOR;
%                               vendor: Philips;
%                               heartbeat detection: ECG, load_from_logfile
%                                                    Philips detected peaks
%                                                    no posthoc-detection
%                               scan_timing:         gradient_log
%                                                    uses gradient data
%                                                    from SCANPHYSLOG-file
%
%                       'RETROICOR' order of RETROICOR expansion taken from
%                       Harvey2008, JRMI28(6), p1337ff.
%                       'scan_timing_from_start'
%                       'manual_peak_select'
%   physio_in       - used as input, only fields related to default_scheme
%                     are overwritten, the others are kept as in physio_in
%
% OUT
%   physio          - the complete physio structure, which can be used in
%                     tapas_physio_main_create_regressors
%
% NOTE
%   All parameters used in the physIO toolbox are defined AND DOCUMENTED in
%   this file. Just scroll down and read through the comments!
%
% EXAMPLE
%   physio = tapas_physio_new('default')
%       OR = tapas_physio_new():
%
%   physio = tapas_physio_new('empty')
%   physio = tapas_physio_new('RETROICOR');
%   physio = tapas_physio_new('manual_peak_select', physio);
%
%   See also tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2013-04-23
% Copyright (C) 2013-2018 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% if not specified differently, create everything empty
if ~nargin
    default_scheme = 'empty';
end

% include sub-folders of code to path as well
pathThis = fileparts(mfilename('fullpath'));
addpath(genpath(pathThis));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modules (Overview)
% Overview over all sub-modules of the PhysIO Toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if nargin >= 2
    save_dir    = physio_in.save_dir;
    log_files   = physio_in.log_files;
    preproc     = physio_in.preproc;
    scan_timing = physio_in.scan_timing;
    sync        = scan_timing.sync;
    sqpar       = scan_timing.sqpar;
    model       = physio_in.model;
    verbose     = physio_in.verbose;
    ons_secs    = physio_in.ons_secs;
else
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save_dir (Module)
    % Directory where output model, regressors and figure-files are saved
    % to; leave empty to use current directory
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Overarching directory, relative to which output files are saved
    save_dir = '';
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% log_files (Module)
    % General physiological log-file information, e.g. file names, sampling
    % rates
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    log_files              = [];
    
    % vendor                Name depending on your MR Scanner system
    %                       'BIDS' - Brain Imaging Data Structure (http://bids.neuroimaging.io/bids_spec.pdf, section 8.6)'
    %                       'Biopac_Txt' - exported txt files from Biopac system (4 columns, [Resp PPU GSR Trigger]'
    %                       'Biopac_Mat' - exported mat files from Biopac system'
    %                       'BrainProducts' - .eeg files from BrainProducts EEG system'
    %                       'Custom'
    %                           'Custom' expects the logfiles (separate files for cardiac and respiratory)'
    %                           to be plain text, with one cardiac (or'
    %                           respiratory) sample per row;'
    %                           If heartbeat (R-wave peak) events are'
    %                           recorded as well, they have to be put'
    %                           as a 2nd column in the cardiac logfile'
    %                           by specifying a 1; 0 in all other rows'
    %                           e.g.:'
    %                           0.2  0'
    %                           0.4  1 <- cardiac pulse event'
    %                           0.2  0'
    %                           -0.3 0'
    %                           NOTE: the sampling interval has to be specified for these files as'
    %                           well (s.b.)'
    %                       'GE'
    %                       'Philips'
    %                       'Siemens'
    %                       'Siemens_Tics' - new Siemens physiological'
    %                           Logging with time stamps in tics'
    %                           (= steps of 2.5 ms since midnight) and'
    %                           extra acquisition (scan_timing) logfile with'
    %                           time stamps of all volumes and slices'
    %                       'Siemens_HCP' - Human Connectome Project (HCP) Physiology Data'
    %                           HCP-downloaded files of  name format  *_Physio_log.txt '
    %                           are already preprocessed into this simple 3-column text format'
    log_files.vendor       = 'Philips';
    
    % Logfile with cardiac data, e.g.
    %   'SCANPHYSLOG<date>.log' (Philips)
    %   '<id>_PAV.ecg'          (Siemens Trio etc. (VB))
    %   '<date>_ECG1-4.log'     (Siemens Prisma etc (VD))
    %   'ECGData_epiRT_<date>'  (GE)
    log_files.cardiac      = '';
    
    % Logfile with respiratory data, e.g. 'SCANPHYSLOG.log';
    % (same as .cardiac for Philips)
    log_files.respiration  = '';
    
    % additional file for relative timing information between logfiles and
    % MRI scans.
    % Currently implemented for 2 cases
    % Siemens:      Enter the first or last DICOM volume of your session here,
    %               The time stamp in the DICOM header is on the same time
    %               axis as the time stamp in the physiological log file
    % Siemens_Tics: log-file which holds table conversion for tics axis to
    %               time conversion
    log_files.scan_timing  = '';
    
    % Sampling interval in seconds (i.e. time between two rows in logfile
    % if empty, default value will be set: 2e-3 for Philips, variable for GE, e.g. 40e-3
    %         1 entry: sampling interval (seconds)
    %         for both cardiac + respiratory log file
    %         2 entries: 1st entry sampling interval (seconds)
    %         for cardiac logfile, 2nd entry for respiratory
    %         logfile
    log_files.sampling_interval = [];
    
    % Time (in seconds) when the 1st scan (or, if existing, dummy) started,
    % relative to the start of the logfile recording;
    % e.g.
    %       [] (empty) to read from explicit acquisition timing info (s.b.)
    %       0 if simultaneous start
    %       10, if 1st scan starts 10
    %       seconds AFTER physiological
    %       recording
    %       -20, if first scan started 20
    %       seconds BEFORE phys recording
    % NOTE:
    %       1. For Philips SCANPHYSLOG, this parameter is ignored, if
    %       scan_timing.sync is set.
    %       2. If you specify an acquisition_info file, leave this
    %       parameter empty or 0 (e.g., for Siemens_Tics, BIDS) since
    %       physiological recordings and acquisition timing are already
    %       synchronized by this information, and you would introduce an
    %       additional shift.
    %
    log_files.relative_start_acquisition = 0;
    
    % Determines which scan shall be aligned to which part of the logfile
    % Typically, aligning the last scan to the end of the logfile is
    % beneficial, since start of logfile and scans might be shifted due
    % to pre-scans
    %
    % NOTE: In all cases, log_files.relative_start_acquisition is
    %       added to timing after the initial alignment to first/last scan
    %
    % 'first'   start of logfile will be aligned to first scan volume
    % 'last'    end of logfile will be aligned to last scan volume
    log_files.align_scan       = 'last';
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% scan_timing (Module)
    % Parameters for sequence timing & synchronization
    % scan_tming.sqpar =    slice and volume acquisition starts, TR,
    %                       number of scans etc.
    % scan_timing.sync =    synchronize phys logfile to scan acquisition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    scan_timing = struct('sqpar', [], 'sync', []);
    
    
    scan_timing.sqpar.Nslices           = [];   % number of slices per volume in fMRI scan
    scan_timing.sqpar.NslicesPerBeat    = [];   % usually equals Nslices, unless you trigger with the heart beat
    scan_timing.sqpar.TR                = [];   % volume repetition time in seconds
    scan_timing.sqpar.Ndummies          = [];   % number of dummy volumes
    
    % number of full volumes saved (volumes in nifti file,
    % usually rows in your design matrix)
    scan_timing.sqpar.Nscans            = [];
    
    % Count of preparation pulses
    % BEFORE 1st dummy scan.
    % Only important, if log_files.scan_align = 'first', since then
    % preparation pulses and dummy triggers are counted and discarded
    % as first scan onset
    scan_timing.sqpar.Nprep             = [];
    
    % time between the acquisition of 2 subsequent
    % slices; typically TR/Nslices or minTR/Nslices,
    % if minimal temporal slice spacing was chosen
    % NOTE: only necessary, if preproc.grad_direction
    % is empty and nominal scan timing is used
    scan_timing.sqpar.time_slice_to_slice  = [];
    
    % slice whose scan onset determines the adjustment of the
    % regressor timing to a particular slice for the whole volume
    % volumes from beginning of run, i.e. logfile,
    % includes counting of preparation gradients
    scan_timing.sqpar.onset_slice       = [];
    
    
    % Method to determine slice acquisition onset times
    % 'nominal'             derive slice acquisition timing from sqpar
    %                       directly
    % 'gradient_log'        derive from logged gradient time courses
    %                       in SCANPHYSLOG-files (Philips only)
    % 'gradient_log_auto'   !!! NOT FUNCTIONAL!!!
    %                       as 'gradient_log' but without defining height/
    %                       spacing thresholds (Philips only)
    % 'scan_timing_log'     uses individual scan timing logfile with time stamps
    %                       specified in log_files.scan_timing
    %                       e.g.,
    %                       *_INFO.log for 'Siemens_Tics' (time stamps for
    %                                       every slice and volume)
    %                       *.dcm (DICOM) for Siemens, is first volume (non-dummy) used
    %                                     in GLM analysis
    %                       *.tsv (3rd column) for BIDS, using the scanner
    %                                     volume trigger onset events
    %                       NOTE:   This setting needs a valid filename to
    %                               entered in log_files.scan_timing
    scan_timing.sync.method = 'gradient_log';
    scan_timing.sync.grad_direction = ''; % 'x', 'y', or 'z';
    
    % if set, sequence timing is calculated
    % from logged gradient time-course along
    % this coordinate axis;
    
    scan_timing.sync.zero     = [];   % gradient values below this value are set to zero;
    
    % should be those which are unrelated to slice acquisition start
    
    % minimum gradient amplitude to be exceeded when a slice scan starts
    scan_timing.sync.slice    = [];
    
    % minimum gradient amplitude to be exceeded when a new volume starts;
    % leave [], if volume events shall be determined as
    % every Nslices-th scan event or via vol_spacing
    scan_timing.sync.vol      = [];
    
    
    % duration (in seconds) from last slice acq to
    % first slice of next volume;
    % leave [], if .vol-threshold shall be used
    scan_timing.sync.vol_spacing          = [];
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% preproc (Module)
    % Preprocessing strategy and parameters for physiological data,
    % starting from raw peripheral measures (preproc.cardiac, preproc.respiration)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    preproc = [];
    
    preproc.cardiac = [];
    
    % Measurement modality of input cardiac signal
    % 'ECG','ECG_raw', or 'OXY'/'PPU' (for pulse oximetry), 'OXY_OLD', [deprecated]
    preproc.cardiac.modality = 'ecg_wifi';
    
    % Filter properties for bandpass-filtering of cardiac signal before peak
    % detection, phase extraction, and other physiological traces
    preproc.cardiac.filter = [];
    
    preproc.cardiac.filter.include = 0; % 1 = filter executed; 0 = not used
    
    % filter type   default: 'cheby2'
    %   'cheby2'    Chebychev Type II filter, use for steep transition from
    %               start to stop band
    %   'butter'    butterworth filter, standard filter with maximally flat
    %               passband (Infinite impulse response), but stronger
    %               ripples in transition band
    preproc.cardiac.filter.type = 'butter';
    
    % [f_min, f_max] frequency interval in Hz of all frequency that should
    %                pass the passband filter
    %                default: [0.3 9] (to remove slow drifts, breathing
    %                                   and slice sampling artifacts)
    %                if empty, no filtering is performed
    preproc.cardiac.filter.passband = [0.3 9];
    
    % [f_min, f_max] frequency interval in Hz of all frequencies, s.th. frequencies
    %                outside this band should definitely *NOT* pass the filter
    %                Default: []
    %                NOTE: only relevant for 'cheby2' filter type
    %                if empty, and passband is empty, no filtering is performed
    %                if empty, but passband exists, stopband interval is
    %                10% increased passband interval
    preproc.cardiac.filter.stopband = [];
    
    % The initial cardiac pulse selection structure: Determines how the
    % majority of cardiac pulses is detected
    % default: 'auto_matched'
    %
    % 'auto_matched'
    %           - auto generation of representative QRS-wave; detection via
    %             maximizing auto-correlation with it
    % 'load_from_logfile'
    %           - from phys logfile, detected R-peaks of scanner
    % 'manual'  - via manually selected QRS-wave for autocorrelations
    % 'load'    - from previous manual/auto run
    preproc.cardiac.initial_cpulse_select.method = 'auto_matched';
    
    % maximum allowed physiological heart rate (in beats per minute)
    % for subject; default: 90 bpm
    % - If set too low, the auto_mathed pulse detection might miss genuine
    %   cardiac pulses
    % - If set too high, it might introduce artifactual pulse events, i.e.
    %   interpreting local maxima within a pulse as new pulse events
    % Adjust this value, if you have a subject with very high heart rate
    % (increase!), or if you have very pronounced local maxima in your wave form
    % (decrease!).
    preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 90;
    
    % file containing reference ECG-peak (variable kRpeak)
    % used for method 'manual' or 'load' [default: not set]
    % if method == 'manual', this file is saved after picking the QRS-wave
    % such that results are reproducible
    % default: initial_cpulse_kRpeakfile.mat
    preproc.cardiac.initial_cpulse_select.file = 'initial_cpulse_kRpeakfile.mat';
    
    % threshold for peak height in z-scored cardiac waveform to find pulse events
    % default: 0.4
    % NOTE: For ECG, might need increase (e.g., 2.0), because of local maximum
    %       of T wave after QRS complex
    preproc.cardiac.initial_cpulse_select.min = 0.4;
    
    % variable saving an example cardiac QRS-wave to correlate with
    % ECG time series
    preproc.cardiac.initial_cpulse_select.kRpeak = [];
    
    % The post-hoc cardiac pulse selection structure: If only few (<20)
    % cardiac pulses are missing in a session due to bad signal quality, a
    % manual selection after visual inspection is possible using the
    % following parameters. The results are saved for reproducibility
    %
    % 'off'     - no manual selection of peaks
    % 'manual'  - pick and save additional peaks manually
    % 'load'    - load previously selected cardiac pulses
    preproc.cardiac.posthoc_cpulse_select.method = 'off';
    
    % filename where cardiac pulses are saved after manual picking
    preproc.cardiac.posthoc_cpulse_select.file = '';
    
    % Suspicious positions of missing or too many cardiac pulses are
    % pre-selected by detecting outliers in histogram of
    % heart-beat-2-beat-intervals
    preproc.cardiac.posthoc_cpulse_select.percentile = 80; % percentile of beat-2-beat interval histogram that constitutes the "average heart beat duration" in the session
    preproc.cardiac.posthoc_cpulse_select.upper_thresh = 60; % minimum exceedance (in %) from average heartbeat duration to be classified as missing heartbeat
    preproc.cardiac.posthoc_cpulse_select.lower_thresh = 60; % minimum reduction (in %) from average heartbeat duration to be classified an abundant heartbeat
    
    
    preproc.respiratory = [];
    
    % [f_min, f_max] frequency interval in Hz of all frequency that should
    %                pass the passband filter. Want to remove high
    %                frequency noise and low frequency drifts, but not
    %                distort e.g. sigh breaths (which can take e.g. 20 s).
    %                default: [0.01 2.0]
    preproc.respiratory.filter.passband = [0.01, 2.0];
    
    % Whether to remove spikes from the raw respiratory trace using a
    % sliding window median filter.
    preproc.respiratory.despike = false;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model (Module)
    % Physiological noise models derived from preprocessed physiological data
    % available models (that can be combined) are:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 'none'        no physiological model is computed; only the read-out
    %               logfile data is read out and saved in physio.ons_secs
    % 'RETROICOR'   as in Glover et al., MRM 44, 2000
    %               order of expansion:  See Harvey et al., JMRI 28, 2008
    % 'HRV'         heart rate variability, as in Chang et al., 2009
    % 'RVT'         respiratory volume time, as in Birn et al., 2006/8
    % 'movement'    realignment parameters, derivatives,
    %               + squared (parameters+derivatives),
    %               (Volterra expansion, see: Friston KJ, Williams S, Howard R, Frackowiak
    %               RS, Turner R. Movement-related effects in fMRI
    %               time-series. Magn Reson Med. 1996;35:346?355.)
    % 'noise_rois'  Principal Components of time series of all voxels in given
    %               regions of localized noise, e.g. CSF, vessels, white
    %               matter
    %               e.g. CompCor: Behzadi, Y., Restom, K., Liau, J., Liu,
    %               T.T., 2007. A component based noise correction method (CompCor) for BOLD and perfusion based fMRI. NeuroImage 37, 90?101. doi:10.1016/j.neuroimage.2007.04.042
    model = [];
    
    % string indicating which regressors shall be orthogonalised;
    % mainly needed, if acquisition was triggered to heartbeat (set to 'cardiac') OR
    % if session mean shall be evaluated (e.g. SFNR-studies, set to 'all')
    % 'n' or 'none'     - no orthogonalisation is performed
    % Possible Values (default: 'none'
    %   'c' or 'cardiac'  - only cardiac regressors are orthogonalised
    %   'r' or 'resp'     - only respiration regressors are orthogonalised
    %   'mult'            - only multiplicative regressors are orthogonalised
    %   'all'             - all physiological regressors are orthogonalised to each other
    %   'RETROICOR'
    %   'HRV'
    %   'RVT'
    %   'movement'
    %   'noise_rois'
    model.orthogonalise = 'none';
    
    % true or false (default)
    % If true, values of the nuisance regressors (R-matrix) will be set to
    % zero (=censored) for time points that are in intervals with
    % unreliable related recordings, i.e.,
    % -  cardiac regressors, if ~c_is_reliable;
    % - respiratory regressors, if ~r_is_reliable
    %
    % NOTE: so far, this is only implemented for instantaneous effect of
    % poor recordings, e.g., the phase estimates leading to RETROICOR
    % regressors.
    % For the convolution models (HRV, RVT), no censoring is performed,
    % since effects of unreliable recordings have long-term effects.
    model.censor_unreliable_recording_intervals = true;
    
    % output file for usage in SPM multiple_regressors GLM-specification
    % either txt-file or mat-file with variable R
    model.output_multiple_regressors = '';
    
    % mat-file where whole physio-structure is saved after finishing main.m
    model.output_physio = '';
    
    % [nScans, nRegressors, nSlices]
    %   output design matrix of confound regressors,
    %   saved in 'multiple_regressors.txt'
    %   or, if multiple slices are specified as onset_slice, in multiple
    %       multiple_regressors_001.txt files, one per specified slice
    model.R = [];
    
    %% RETROICOR (Model): Glover et al. 2000
    % Retrospective image correction method, based on Fourier expansion of
    % cardiac and respiratory phase, plus multiplicative interaction terms
    % (Harvey et al., 2008)
    
    model.retroicor.include = 1; % 1 = included; 0 = not used
    % natural number, order of cardiac phase Fourier expansion
    model.retroicor.order.c = 3;
    
    % natural number, order of respiratory phase Fourier expansion
    model.retroicor.order.r = 4;
    
    % natural number, order of cardiac-respiratory-phase-interaction Fourier expansion
    model.retroicor.order.cr = 1;
    
    
    %% RVT (Model): Respiratory Volume per time model , Birn et al., 2006/8
    model.rvt.include = 0;
    
    % Whether to estimate RVT from the Hilbert transform ('hilbert') or via
    % peak detection ('peaks').
    model.rvt.method = 'hilbert';
    
    % one or multiple delays (in seconds) can be specified to shift
    % canonical RVT response function from Birn et al., 2006 paper
    % Delays e.g. 0, 5, 10, 15, and 20s (Jo et al., 2010 NeuroImage 52)
    model.rvt.delays = 0;
    
    
    %% HRV (Model): Heart Rate variability, Chang et al., 2009
    model.hrv.include = 0;
    
    % one or multiple delays (in seconds) can be specified to shift
    % canonical HRV response function from Chang et al., 2009 paper
    % Delays e.g. 0:6:24s (Shmueli et al, 2007, NeuroImage 38)
    model.hrv.delays = 0;
    
    
    %% noise_rois (Model): Anatomical Component Correction, Behzadi et al, 2007
    % Principal Components of time series of all voxels in given regions
    % of localized noise, e.g. CSF, vessels, white matter
    %
    % e.g. CompCor: Behzadi, Y., Restom, K., Liau, J., Liu,
    % T.T., 2007. A component based noise correction method (CompCor)
    % for BOLD and perfusion based fMRI. NeuroImage 37, 90-101.
    % doi:10.1016/j.neuroimage.2007.04.042
    model.noise_rois.include = 0;
    
    % cell of preprocessed fMRI nifti/analyze files, from which time series
    % shall be extracted
    model.noise_rois.fmri_files = {};
    
    % cell of Masks/tissue probability maps characterizing where noise resides
    model.noise_rois.roi_files = {};
    
    % Single threshold or vector [1, nRois] of thresholds to be applied to mask files to decide
    % which voxels to include (e.g. a probability like 0.99, if roi_files
    % are tissue probability maps)
    model.noise_rois.thresholds = 0.9;
    
    % Single number or vector [1, nRois] of number of voxels to crop per ROI
    % default: 0
    model.noise_rois.n_voxel_crop = 0;
    
    % Single number or vector [1, nRois] of numbers
    % integer >=1:      number of principal components to be extracted
    %                   from all voxel time series within each ROI
    % float in [0,1[    choose as many components as needed to explain this
    %                   relative share of total variance, e.g. 0.99 =
    %                   add more components, until 99 % of variance explained
    % NOTE: Additionally, the mean time series of the region is also
    % extracted
    model.noise_rois.n_components = 1;
    
    % Noise ROIs volumes must have the same geometry as the functional time series.
    % It means same affine transformation(space) and same matrix(voxel size)
    % Possible values:
    % 'Yes' or 1/true (default)
    %   Coregister : Estimate & Reslice will be performed on the noise NOIs,
    %   so their geometry (space + voxel size) will match the fMRI volume.
    % 'No' or 0 or false
    %   Geometry will be tested:
    %   1) If they match, continue
    %   2) If they don't match, perform a Coregister : Estimate & Reslice as fallback
    model.noise_rois.force_coregister = 1;
    
    %% movement (Model): Regressor model 6/12/24, Friston et al. 1996
    % Also: sudden movement exceedance regressors
    
    model.movement.include = 1;
    model.movement.file_realignment_parameters = '';
    
    % 0 = no realignment parameters included
    % 6 = rotation/translation parameters
    % 12 = + derivatives
    % 24 = + squared parameters and derivatives, corresponding to a
    %        Volterra expansion V_t, V_t^2, V_(t-1), V_(t-1)^2
    model.movement.order = 6;
    
    % Censoring outlier threshold;
    % Threshold, above which a stick (''spike'') regressor is created for
    % corresponding outlier volume exceeding threshold'
    %
    % The actual setting depends on the chosen thresholding method:
    % 'MAXVAL'   -  [1,1...6] max translation (in mm) and rotation (in deg) threshold
    %                recommended: 1/3 of voxel size (e.g., 1 mm)
    %                default: 1 (mm)
    %                1 value   -> used for translation and rotation
    %                2 values  -> 1st = translation (mm), 2nd = rotation (deg)
    %                6 values  -> individual threshold for each axis (x,y,z,pitch,roll,yaw)
    % 'FD'       -   [1,1] framewise displacement (in mm)
    %                default: 0.5 (mm)
    %                recommended for subject rejection: 0.5 (Power et al., 2012)
    %                recommended for censoring: 0.2 (Power et al., 2015)
    % 'DVARS'    -   [1,1] in percent BOLD signal change
    %                recommended for censoring: 1.4 % (Satterthwaite et al., 2013)
    model.movement.censoring_threshold = 0.5;
    
    % Censoring method used for thresholding
    % Motion Censoring ('spike' regressors for motion-corrupted volumes)
    % 1 stick regressor for outlier volume with respect to a certain
    % quality criterion will be created, using one of these methods:
    %
    %   'None'      - no motion censoring performed
    %   'MAXVAL'    - thresholding (max. translation/rotation)
    %   'FD''       - frame-wise displacement (as defined by Power et al., 2012)
    %                 i.e., |rp_x(n+1) - rp_x(n)| + |rp_y(n+1) - rp_y(n)| + |rp_z(n+1) - rp_z(n)|
    %                       + 50 mm *(|rp_pitch(n+1) - rp_pitch(n)| + |rp_roll(n+1) - rp_roll(n)| + |rp_yaw(n+1) - rp_yaw(n)|
    %                 where 50 mm is an average head radius mapping a rotation into a translation of head surface
    %   'DVARS'     - root mean square over brain voxels of
    %                 difference in voxel intensity between consecutive volumes
    %                 (Power et al., 2012))
    model.movement.censoring_method = 'FD';
    
    % output structure, if censoring is used
    model.movement.censoring = [];
    
    %% other (Model): Additional, pre-computed nuisance regressors
    % To be included in design matrix as txt or mat-file (variable R)
    model.other.include = 0;
    model.other.input_multiple_regressors = '';
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% verbose (Module)
    % Verbosity of Toolbox, i.e. how many figures shall be generated to
    % visualize the workflow and save it (to graphics file(s))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    verbose = [];
    
    % verbosity levels:
    %-1 = no text or graphics output (text saved in verbose.process_log)
    % 0 = no graphical output;
    % 1 = (default) main plots : Fig 1: gradient scan timing (if selected) ;
    %                            Fig 2: heart beat/breathing statistics & outlier;
    %                            Fig 3: final multiple_regressors matrix
    % 2 = debugging plots        for setting up new study or if Fig 2 had
    %                            outliers
    %                            Fig 1: raw phys logfile data
    %                            Fig 2: gradient scan timing (if selected)
    %                            Fig 3: cutout interval of logfile for
    %                            regressor creation (including scan timing
    %                            and raw phys data)
    %                            Fig 4: heart beat/breathing statistics & outlier;
    %                            Fig 5: time course of all sampled RETROICOR
    %                                   regressors
    %                            Fig 6: final multiple_regressors matrix
    %
    % 3 = all plots
    %                            Fig 1: raw phys logfile data
    %                            Fig 2: gradient scan timing (if selected)
    %                            Fig 3: Slice assignment to volumes
    %                            Fig 4: cutout interval of logfile for
    %                            regressor creation (including scan timing
    %                            and raw phys data)
    %                            Fig 5: heart beat/breathing statistics & outlier;
    %                            Fig 6: cardiac phase data of all slices
    %                            Fig 7: respiratory phase data and
    %                                   histogram transfer function
    %                            Fig 8: time course of all sampled RETROICOR
    %                                   regressors
    %                            Fig 9: final multiple_regressors matrix
    verbose.level = 1;
    
    % stores text outputs of PhysIO Toolbox processing, e.g. warnings about missed
    % slice triggers, peak height etc.
    verbose.process_log = cell(0,1);
    
    % [1, nFigs] vector; collecting of all generated figure handles during a run of tapas_physio_main_create_regressors
    verbose.fig_handles = zeros(1,0);
    
    % file name (including extension) where to print all physIO output
    % figures to.
    % e.g. 'PhysIO_output.ps' or 'PhysIO_output.jpg'
    %
    % The specified extension determines how the figures will be saved
    %     .ps - all figures are saved to the same, multi-page postscript-file
    %     .fig, .tiff,  .jpg
    %         - one file is created for each figure, appended by a figure
    %         index, e.g. 'PhysIO_output_fig01.jpg'
    verbose.fig_output_file = '';
    
    % NOT IMPLEMENTED YET
    %  If true, plots are performed in tabs of SPM graphics window
    %  TODO: implement via [handles] = spm_uitab(hparent,labels,callbacks,...
    %                                           tag,active,height,tab_height)
    %
    verbose.use_tabs = false;
    
    % show / save / close figures
    % Booleans to control figure visibility and saving/closing options.
    
    % show_figs: If true, all created figures will be visible. If false, sets figure visibility to 'off', which
    % leaves the possibility of saving the figure to file without displaying it.
    verbose.show_figs = true;
    % save_figs: If true, all created figures will be saved using specified fig_output_file as filename
    verbose.save_figs = false;
    % close_figs: If true, will close all open figs after saving them. Only used if save_figs is true.
    verbose.close_figs = false;
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ons_secs (Module, output only)
    % Output structure for all read or computed  time-dependent variables,
    % i.e. onsets, specified in seconds
    % NOTE: all elements but .raw are cropped to the acquisition window of
    % the session
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ons_secs                     = [];
    
    % read-in data
    ons_secs.t              	 = [];  % time vector corresponding to c and r
    ons_secs.t_start             = [];  % offset time when logfile started, such that t(1)=0 and t contains relative times
    ons_secs.c              	 = [];  % raw cardiac waveform (ECG or PPU)
    ons_secs.r              	 = [];  % raw respiration amplitude time course
    ons_secs.c_scaling           = 1;   % stores scaling factor for cardiac data
    % after normalization
    ons_secs.r_scaling           = 1;   % stores scaling factor for respiratory data
    % after normalization
    
    % flags for detected unreliable intervals of physiological recording
    ons_secs.c_is_reliable       = [];  % 1 for all time points where cardiac recording is reliable, 0 elsewhere (e.g. high noise, too low/high heartrates)
    ons_secs.r_is_reliable       = [];  % 1 for all time points, where respiratory recording is reliable; 0 elsewhere (e.g. constant amplitude through detachment/clipping)
    
    % processed elements cardiac pulse detection and phase estimations
    ons_secs.cpulse         	 = [];  % onset times of cardiac pulse events (e.g. R-peaks)
    ons_secs.fr                  = [];  % filtered respiration amplitude time series
    ons_secs.c_sample_phase      = [];  % phase in heart-cycle when each slice of each volume was acquired
    ons_secs.r_sample_phase      = [];  % phase in respiratory cycle when each slice of each volume was acquired
    ons_secs.hr                  = [];  % [nScans,1] estimated heart rate at each scan
    ons_secs.rvt                 = [];  % [nScans,1] estimated respiratory volume per time at each scan
    
    % statistical info about physiological data
    ons_secs.c_outliers_high     = [];  % onset of too long heart beats
    ons_secs.c_outliers_low      = [];  % onsets of too short heart beats
    ons_secs.r_hist              = [];  % histogram of breathing amplitudes
    
    % scan timing parameters
    ons_secs.svolpulse      	 = [];  % [Nscans x 1] onset times of volume scan events
    ons_secs.spulse         	 = [];  % [Nscans*Nslices x 1] onset times of slice (incl. volume) scan events
    ons_secs.spulse_per_vol 	 = [];  % cell(Nscans,1), as spulse, holding slice scan events sorted by volume
    
    % uncropped parameters
    ons_secs.raw            	 = [];  % raw read-in version of the whole structure, before any cropping
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-defined model configurations
%  Certain modeling choices are very common, and the accompanying
%  parameter settings can be generated by calling this constructor with a
%  default scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch default_scheme
    case 'RETROICOR'
        model.type = 'RETROICOR';
        model.order = struct('c',3,'r',4,'cr',1, 'orthogonalise', 'none');
    case 'redetect_peaks_from_logfile'
        preproc.cardiac.initial_cpulse_select.method = 'manual'; % 'load_from_logfile', 'manual', 'load'
        preproc.cardiac.initial_cpulse_select.file = 'kRpeak.mat';
        preproc.cardiac.initial_cpulse_select.min = 1;
        preproc.cardiac.initial_cpulse_select.kRpeak = [];
    case 'manual_peak_select'
        preproc.cardiac.posthoc_cpulse_select.method = 'manual'; % 'off', 'manual' or 'load',
        preproc.cardiac.posthoc_cpulse_select.file = 'posthoc_cpulse.mat';
        preproc.cardiac.posthoc_cpulse_select.percentile = 80;
        preproc.cardiac.posthoc_cpulse_select.upperThresh = 60;
        preproc.cardiac.posthoc_cpulse_select.lowerThresh = 30;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assemble output (all PhysIO-modules)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

physio.save_dir     = save_dir;
physio.log_files    = log_files;
physio.scan_timing  = scan_timing;
physio.preproc      = preproc;
physio.model        = model;
physio.verbose      = verbose;
physio.ons_secs     = ons_secs;

% Call functions for specific initial value settings (e.g. 3T Philips system)
switch default_scheme
    case 'Philips'
        physio = tapas_physio_new_philips(physio);
end

physio.version = tapas_physio_version();