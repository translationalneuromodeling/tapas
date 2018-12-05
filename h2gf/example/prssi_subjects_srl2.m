function [ details, paths ] = prssi_subjects_srl2( id )
%PRSSI_SUBJECTS_SRL2 Function that sets all filenames and paths
%   IN:     EITHER (for quering both general and subject-specific paths:
%           id                  - the subject number as a string, e.g. '0352'
%           options (optional)  - the struct that holds all analysis options
%           OR (for only quering general paths & files):
%           options     - the struct that holds all analysis options 
%   OUT:    details     - a struct that holds all filenames and paths for
%                       the subject with subject number id
%           paths       - a struct that holds all paths and config files
%                       that are not subject-specific


%-- check input ------------------------------------------------------------
if isstruct(id)
    options = id;
    id = 'dummy';
elseif ischar(id) && nargin < 2
    options = prssi_set_analysis_options_srl2;
end

if ismember(id, options.srl_short.subjectIDs)
    options.part = 'srl_short';
elseif ismember(id, options.srl_long.subjectIDs)
    options.part =  'srl_long';
elseif strcmp(id, 'dummy')
    % do nothing, keep options.part as is 
else
    error('Could not determine whether id belongs to short or long part');
end

%-- general paths and files -----------------------------------------------------------------------%
paths.confroot      = fullfile(options.workdir, 'config');
paths.designroot    = fullfile(options.workdir, 'design');
paths.qualityroot   = fullfile(options.resultsdir, options.part, 'quality');
paths.qualityfold   = fullfile(paths.qualityroot, options.preproc.eyeblinktreatment);
paths.statsroot     = fullfile(options.resultsdir, options.part, 'stats');   
paths.statsfold     = fullfile(paths.statsroot, options.stats.mode, options.preproc.eyeblinktreatment);   
paths.erproot       = fullfile(options.resultsdir, options.part, 'erp', options.erp.type);
paths.erpfold       = fullfile(paths.erproot, options.erp.type);     

% config files
paths.montage       = fullfile(paths.confroot, ['SNS_montage.mat']);
paths.tnufile     = fullfile(paths.confroot, '/waveguard_128coords8_SFP_corr2.sfp');
paths.channeldef    = fullfile(paths.confroot, 'sns_eeg_channels.mat'); %was details.eegchannels
paths.trialdef      = fullfile(paths.confroot, ['/prssi_trialdef_' ...
                        options.preproc.trialdef '.mat']);                        
                        
% % design files
% paths.paradigm      = fullfile(paths.designroot, ['paradigm_' options.stats.priors '.mat']);
% paths.designFile    = fullfile(paths.designroot, ['design_' options.stats.design '_' ...
%                             options.stats.priors '.mat']);  
%                         
% logging 2nd level analyses and quality check
paths.logfile       = fullfile(options.workdir, options.part, 'secondlevel.log');
paths.trialstatstab = fullfile(paths.qualityfold, [options.preproc.eyeblinktreatment, '_table_trial_stats.mat']);
paths.trialstatsfig = fullfile(paths.qualityfold, [options.preproc.eyeblinktreatment, '_overview_trial_stats']);
                        
% currently, these are the standard SPM fiducial positions
paths.spmfid.labels      = {'NAS'; 'LPA'; 'RPA'};
paths.spmfid.data        = [  1,    85,    -41; ...
                           -83,   -20,    -65; ...
                            83,   -20,    -65];

%-- subject-specific options, paths and files -----------------------------------------------------%
% differences in file endings
% --> not the case for this study

% EB detection threshold
switch id
    case {'dumm', 'dum1'}
        details.eyeblinkthreshold = 2;
    case {'dum3', 'dum4'}
        details.eyeblinkthreshold = 5;
    otherwise
        details.eyeblinkthreshold = options.preproc.eyeblinkthreshold;
end

% bad channels before EB confound estimation
switch id
    case 'dum2'
        details.preclean.badchannels = 39;
    otherwise
        details.preclean.badchannels = [];
end

%% names
% differences in file endings
switch id
    case {'9997', '6666'}
        rawsuffix = '_task';
    otherwise 
        rawsuffix = '';
end

details.subjname        = ['PRSSI_' id];
details.subjname_MRI    = ['TNU_PRSSI_' id];
details.subjfolder      = ['TNU_PRSSI_' id];

disp(details.subjname);
disp('******************');

subj_can_n = {'PRSSI_0303', 'PRSSI_0304'};
b = strcmp(details.subjname,subj_can_n);
B = any(b);
if B == 1
    details.rawfilename = ['PRSSI_' options.task '_' id];
elseif strcmp(details.subjname, 'PRSSI_0330')
    details.rawfilename = [details.subjname '_' options.task];
else
    details.rawfilename = [details.subjname '_' options.task rawsuffix];
end

details.prepfilename  = [details.subjname '_' options.task '_preproc'];
details.artfname      = ['ebbfdfMspmeeg_' details.subjname '_' options.task rawsuffix];
details.redefname     = ['redef_' details.subjname '_' options.erp.type];
details.avgname       = ['m' details.subjname '_' options.erp.type];
details.filname       = ['fm' details.subjname '_' options.erp.type];
details.erpname       = [details.subjname '_ERP_' options.erp.type];
details.diffname      = [details.subjname '_diffWaves_' options.erp.type];

%% directories

details.rawroot     = fullfile(options.rawdir, details.subjfolder, 'eegdata');
details.mriroot     = fullfile(options.mriroot, 'results_eeg1_srl', details.subjfolder);

details.subjroot    = fullfile(options.resultsdir, details.subjfolder);
details.preproot    = fullfile(details.subjroot, 'spm_prep', options.preproc.eyeblinktreatment);
details.erproot     = fullfile(details.subjroot, 'spm_erp', options.preproc.eyeblinktreatment);

details.erpfold     = fullfile(details.erproot, options.erp.type);

details.designroot  = fullfile(details.subjroot, 'design');
details.statsroot   = fullfile(details.subjroot, 'spm_stats', options.preproc.eyeblinktreatment);
% details.statsfold   = fullfile(details.statsroot, options.stats.priors);
details.spmresuroot = [details.statsroot '/' options.conversion.convPrefix '/'];

details.behavroot   = fullfile(options.rawdir, details.subjfolder, '/behavior/srl/');
details.behavrootresults = fullfile(details.subjroot, '/behavior/srl/');

details.glmrootresults = fullfile(details.subjroot, '/glm/srl/');

switch options.conversion.mode
    case 'modelbased'
        details.convroot = fullfile(details.preproot, ...
            [options.conversion.space '_' options.conversion.convPrefix '_' details.prepfilename]);
    case 'ERPs'
        details.convroot = fullfile(details.erpfold, ...
            [options.conversion.convPrefix '_' details.erpname]);
    case 'diffWaves'
        details.convroot = fullfile(details.erpfold, ...
            [options.conversion.convPrefix '_' details.diffname]);        
end


%% files
% log files
details.dirlogfile = [details.subjroot '/logfile/'];

details.logfile_behavFirst      = fullfile(details.dirlogfile, ...
                            [details.subjname,'_', options.task '_behavFirst.log']);
details.logfile_EEGFirst        = fullfile(details.dirlogfile, ...
                            [details.subjname,'_', options.task '_EEGFirst.log']);
details.logfile_behavFirst      = fullfile(details.dirlogfile, ...
                            [details.subjname,'_', options.task '_behavFirst.log']);
details.logfile                 = fullfile(details.dirlogfile, ...
                            [details.subjname,'_', options.task '_prep.log']);

% mri files
details.mrifile     = fullfile([options.mriroot, '/structural/r' ,details.subjname_MRI, '.nii']);
details.mri_reo     = fullfile([options.mriroot, '/structural/' ,details.subjname_MRI, '.nii']);
details.mri_raw     = fullfile([options.mriroot, '/structural/backup/' ,details.subjname_MRI, '.nii']);

% individual SPM fiducial positions
disp([details.mriroot '/structural/backup/c',details.subjname,'_fid.txt']);
fid = fopen([details.mriroot '/structural/backup/c',details.subjname,'_fid.txt']);
if fid == -1
    disp('dummy subject');
else
    M = textscan(fid,'%f32%f32%f32');
    details.fiducials = [M{1} M{2} M{3}];
    details.fid.data = details.fiducials(1:3,:);
    details.fid.labels = {'NAS'; 'LPA'; 'RPA'};
    details.fiducialtxt     = fullfile(details.mriroot, '/structural/backup/c', details.subjname,'_fid.txt');
end
% digitization
details.elecfile    = fullfile(details.subjroot, 'digit', [details.subjname '_digit.elc']);
details.correlec    = fullfile(details.subjroot, 'digitization', [details.subjname '_corrected.sfp']);           
details.tempelec    = fullfile(options.workdir, 'config', 'waveguard_128coords8_SFP_corr2.sfp');

% eeg files
subj_can_n = {'PRSSI_0303', 'PRSSI_0304'};
b = strcmp(details.subjname,subj_can_n);
B = any(b);
if B == 1
    details.rawfile = fullfile(details.rawroot, ['PRSSI_' options.task '_' id '.eeg']);
elseif strcmp(details.subjname, 'PRSSI_0330')
    details.rawfile = fullfile(details.rawroot, [details.subjname '_' options.task '_2.eeg']);
else
    details.rawfile = fullfile(details.rawroot, [details.subjname '_' options.task, '.eeg']);
end

details.prepfile    = fullfile(details.preproot, [details.prepfilename '.mat']);
details.artffile    = fullfile(details.preproot, [details.artfname '.mat']);
details.trialstats      = fullfile(details.preproot, [details.subjname '_trialStats.mat']);
details.ebfile          = fullfile(details.preproot, ['fEBbfdfMspmeeg_' details.subjname '.mat']);

details.redeffile       = fullfile(details.erpfold, [details.redefname '.mat']);
details.avgfile         = fullfile(details.erpfold, [details.avgname '.mat']);
details.filfile         = fullfile(details.erpfold, [details.filname '.mat']);
details.erpfile         = fullfile(details.erpfold, [details.erpname '.mat']);
details.difffile        = fullfile(details.erpfold, [details.diffname '.mat']);

% design
% details.design          = paths.designFile;
% details.eyeDesign       = fullfile(details.subjroot, 'design', ...
%                             ['design_EBcorr_' options.preproc.eyeblinktreatment '_' ...
%                             options.stats.design '_' options.stats.priors '.mat']); 
% details.subjectDesign   = fullfile(details.subjroot, 'design', ...
%                             ['design_' options.preproc.eyeblinktreatment '_' ...
%                             options.stats.design '_' options.stats.priors '.mat']);   

% stats
details.spmfile         = fullfile(details.statsroot, 'SPM.mat');
details.erphfile        = fullfile(details.convroot, [details.subjname '_erph.mat']);


% conv - conditions
switch options.conversion.mode
    case 'modelbased'
        details.convCon{1} = 'Outcomes';
    case 'diffWaves' 
        switch options.erp.type
            case 'roving'
                details.convCon{1} = 'mmn';
        end
    case 'ERPs'
        switch options.erp.type
            case 'CV'
                details.convCon{1} = 'Outcome_correct';
                details.convCon{2} = 'Outcome_wrong';
        end
end

for i = 1: length(details.convCon)
    details.convfile{i} = fullfile(details.convroot, ['condition_' details.convCon{i} '.nii,']);
    details.smoofile{i} = fullfile(details.convroot, ['smoothed_condition_' details.convCon{i} '.nii,']);
end

% figures
details.ebdetectfig     = fullfile(details.preproot, [details.subjname '_eyeblinkDetection.fig']);
% only needed for EB rejection:
details.eboverlapfig    = fullfile(details.preproot, [details.subjname '_blinktrial_overlap.fig']);    
% only needed for EB correction:
details.ebspatialfig    = fullfile(details.preproot, [details.subjname '_eyeblinkConfounds.fig']);
% details.componentconfoundsfigure = [details.preproot '/',details.subjname,'_compconf.fig'];
details.ebcorrectfig    = fullfile(details.preproot, [details.subjname '_eyeblinkCorrection.fig']);
details.coregdatafig    = fullfile(details.preproot, [details.subjname '_coregistration_data.fig']);
details.coregmeshfig    = fullfile(details.preproot, [details.subjname '_coregistration_mesh.fig']);

details.regressorplots  = fullfile(details.designroot, [details.subjname '_' ...
    options.preproc.eyeblinktreatment '_regressor_check']);
details.firstlevelmaskfig = fullfile(details.statsroot, [details.subjname '_firstlevelmask.fig']);

details.erpfig          = fullfile(details.erpfold, [details.erpname '_' options.erp.plotchannel '.fig']);

% details.checkmesh = [details.preproot '/',details.subjname,'_checkmesh.fig'];
% details.checkdatareg = [details.preproot '/',details.subjname,'_checkdatareg.fig'];
details.checkforward = [details.preproot '/',details.subjname,'_checkforward.fig'];


%% batches
details.reorient = [options.workdir '/batches/reorient_struct.mat'];
details.coreg    = [options.workdir '/batches/coreg_resl.mat'];


%% figure headmodel
details.headmodel1    = fullfile(details.preproot, ...
                            [details.subjname ...
                            '_headmodel1.fig']);
details.headmodel2    = fullfile(details.preproot, ...
                            [details.subjname ...
                            '_headmodel2.fig']);
%% experimental design files
details.design_epsilon = [details.glmrootresults '/PM.mat'];
details.design_da = [details.glmrootresults '/PM_da.mat'];
details.design_epsilon_hgfv5_1 = [details.glmrootresults '/PM_hgfv5_1.mat'];
details.design_da_hgfv5_1 = [details.glmrootresults '/PM_da_hgfv5_1.mat'];
details.design_epsilon_hgfv5_1_eye_corrected = [details.glmrootresults '/PM_hgfv5_1_eye_corrected.mat'];
details.design_da_hgfv5_1_eye_corrected = [details.glmrootresults '/PM_da_hgfv5_1_eye_corrected.mat'];
details.design_epsilon_hgfv5_1_all_corrected = [details.glmrootresults '/PM_hgfv5_1_all_corrected.mat'];
details.design_da_hgfv5_1_all_corrected = [details.glmrootresults '/PM_da_hgfv5_1_all_corrected.mat'];
details.design_epsilon_hgfv5_1_artf_corrected = [details.glmrootresults '/PM_hgfv5_1_cond_artf_corrected.mat'];
details.design_da_hgfv5_1_artf_corrected = [details.glmrootresults '/PM_da_hgfv5_1_cond_artf_corrected.mat'];


details.irr = [details.behavrootresults '/HGF_1_fixom_v1_0/SRL.Re.mat'];
details.nTrials = 160;

% %% EEG preprocessing files
% details.badchannels     = fullfile(details.preproot, [details.subjname '_badchannels.mat']);
% details.ntrialscorr     = fullfile(details.preproot, [details.subjname '_nTrialsCorr.mat']);
% details.artefacts       = fullfile(details.preproot, [details.subjname '_nArtefacts.mat']);
% details.idxeyeartefacts = fullfile(details.preproot, [details.subjname '_idxEyeartefacts.mat']);
% details.numeyeartefacts = fullfile(details.preproot, [details.subjname '_nEyeartefacts.mat']);
% details.ntrialsgood     = fullfile(details.preproot, [details.subjname '_nTrialsGood.mat']);
% details.numArtefacts    = fullfile(details.preproot, [details.subjname '_numArtefacts.mat']);
% %% Eyeblink correction
% details.eyeblinks1       = fullfile(details.preproot, ...
%                             [details.subjname '_nEyeblinksV.mat']);
% details.eyeblinks2       = fullfile(details.preproot, ...
%                             [details.subjname '_nEyeblinksH.mat']);  
% details.eyeblinkfigV    = [details.preproot details.subjname '_eyeblinkDetectionV.fig'];
% details.eyeblinkfigH    = [details.preproot details.subjname '_eyeblinkDetectionH.fig'];
% details.eyeblinkfig2V    = [details.preproot details.subjname '_eyeblinkComponent2V.fig'];
% details.eyeblinkfig2H    = [details.preproot details.subjname '_eyeblinkComponent2H.fig'];
% details.eyeblinkfig1    = fullfile(details.preproot, ...
%                             [details.subjname ...
%                             '_eyeblinkDetection.fig']);
% details.eyeblinkfig2    = [details.preproot details.subjname '_eyeblinkConfounds.fig'];
% details.overlapfig      = fullfile(details.preproot, [details.subjname ...
%                             '_blinktrial_overlap.fig']);  


%% create folders
mkdir(details.subjroot);
mkdir([details.subjroot '/glm/srl']);
mkdir(details.dirlogfile);
mkdir([details.subjroot '/eegdata/' options.preproc.version]);
mkdir(details.statsroot);

end
