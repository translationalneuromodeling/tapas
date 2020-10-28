function options = prssi_set_analysis_options_srl1
% PRSSI_SET_ANALYSIS_OPTIONS_SRL2 Analysis options function for DPRST project
%   Run this function from the working directory of the analysis (the
%   project folder).
%   IN:     -
%   OUT:    options     - a struct with all analysis parameters for the
%                       different steps of the analysis.

%% code and task
options.maincodedir = 'C:\Users\Sandra Iglesias\Documents\Documents\code\GitTNU\tapas\h2gf\example';
options.maindatadir = 'D:\PRSSI\EEG1\';
options.workdir     = options.maincodedir;
% options.workdir = fullfile(options.maincodedir, 'prj', 'code_srl_SSP');
options.task = 'SRL'; %or SSL
options.part = 'srl_long'; %or 'srl_short'

%% data paths
options.rawdir          = fullfile(options.maindatadir, 'raw');
options.resultsdir      = fullfile(options.maindatadir, 'results');
options.mriroot     = fullfile(options.maindatadir, 'mri2');

%% subject
options = prssi_get_part_specific_options(options);

%-- preparation ----------------------------------------------------------%
options.prepare.subjectIDs  = options.subjectIDs;
options.prepare.overwrite   = 0; % whether to overwrite any previous prep

%-- modeling -------------------------------------------------------------%
options.model.subjectIDs    = options.subjectIDs;
options.model.overwrite     = 0; % whether to overwrite any previous model

%-- preprocessing --------------------------------------------------------%
options.preproc.subjectIDs  = options.subjectIDs;
options.preproc.overwrite   = 1; % whether to overwrite any prev. prepr
options.preproc.keep        = 1;

options.preproc.rereferencing       = 'average'; % average (i.e., no reref)
options.preproc.keepotherchannels   = 1;
options.preproc.version             = 'spm_prep_low40_250_srl2';

options.preproc.lowpassfreq         = 40;
options.preproc.highpassfreq        = 0.5;
options.preproc.downsamplefreq      = 250;

options.preproc.trialdef            = 'SRL'; % or CW
options.preproc.trlshift            = 0; % no shift
options.preproc.epochwin            = [0 500];
options.preproc.baselinecorrection  = 0; % no baseline correction

%% eyeblink correction parameters
options.preproc.eyeblinktreatment   = 'ssp'; % 'reject', 'ssp'
options.preproc.mrifile             = 'subject'; % might be template
options.preproc.eyeblinkchannels    = {'VEOG'};
options.preproc.eyeblinkthreshold   = 3; % for SD thresholding: in standard deviations, for amp in uV
% subj_eye = {'TNU_PRSSI_0352', 'TNU_PRSSI_0370','TNU_PRSSI_0382'}%,   'TNU_PRSSI_0315'};
% e = strcmp(options.preproc.fullsubjectID,subj_eye);
% E = any(e);
% if E == 1
%     options.preproc.eyeblinkthreshold   = 2;
% end
options.preproc.windowForEyeblinkdetection = 3; % first event of interest (and optionally last)
% options.preproc.eyeblinkdetection   = 'sdThresholding'; %commented for
% newer EB detection

options.preproc.eyeblinkmode        = 'eventbased'; % uses EEG triggers for trial onsets
options.preproc.eyeblinkwindow      = 0.5; % in s around blink events
options.preproc.eyeblinktrialoffset = 0; % 0.1 in s: EBs won't hurt <100ms after tone onset
options.preproc.eyeblinkEOGchannel  = 'VEOG'; % EOG channel (name/idx) to plot

options.preproc.eyeblinkEEGchannel  = []; % here we don't plot any EEG channel
options.preproc.eyebadchanthresh    = 0.4; % prop of bad trials due to EBs

options.preproc.eyeconfoundcomps    = 1; %nr EB confounds for SSP
options.preproc.eyecorrectionchans  = {'Fp1', 'Fz', 'AF8', 'T7', 'Oz'};
options.preproc.preclean.doFilter           = true;
options.preproc.preclean.lowPassFilterFreq  = 10;
options.preproc.preclean.doBadChannels      = false;
options.preproc.preclean.doRejection        = true;
options.preproc.preclean.badtrialthresh     = 500;
options.preproc.preclean.badchanthresh      = 0.5;
options.preproc.preclean.rejectPrefix       = 'cleaned_';

options.preproc.badtrialthresh      = 75; % in microVolt
options.preproc.badchanthresh       = 0.2; % prop of bad trials (artefacts)


options.preproc.artifact.badtrialthresh = 500; % in microVolt


%% ERP
options.erp.subjectIDs 	= options.subjectIDs;
options.erp.overwrite   = 0; % whether to overwrite any previous erp

options.erp.type = 'CW'; %CW: correct-wrong

switch options.erp.type
    case 'CW'
        options.erp.conditions = {'correct', 'wrong'};
    case {'SRL'}
        options.erp.conditions = {'Outcome'};
end
options.erp.plotchannel = 'Fz';
options.erp.averaging   = 's'; % s (standard), r (robust)
switch options.erp.averaging
    case 'r'
        options.erp.addfilter = 'f';
    case 's'
        options.erp.addfilter = '';
end

options.erp.contrastWeighting   = 1;
options.erp.contrastPrefix      = 'diff_';
options.erp.contrastName        = 'erp_srl';

%-- conversion2images ----------------------------------------------------%
options.conversion.subjectIDs       = options.subjectIDs;
options.conversion.overwrite        = 0; % whether to overwrite prev. conv.

options.conversion.mode             = 'modelbased'; %'ERPs', 'modelbased', 
                                                    %'diffWaves'
options.conversion.space            = 'sensor';
options.conversion.convPrefix       = 'whole'; % whole, early, late, ERP
options.conversion.convTimeWindow   = [0 500];
options.conversion.smooKernel       = [16 16 0];

%% stats
options.stats.subjectIDs            = options.subjectIDs;
options.stas.overwrite              = 0; % whether to overwrite any previous erp
options.stats.mode                  = 'modelbased';        % 'modelbased', 'ERP'
%% hgf version
options.stats.design                = 'HGF_low40_250_srl2';
options.stats.design_da             = 'HGF_low40_250_da_srl2';
options.stats.design_hgfv4_1        = 'HGF_low40_250_srl2_hgfv4_1';
options.stats.design_da_hgfv4_1     = 'HGF_low40_250_da_srl2_hgfv4_1';
options.stats.design_hgfv5_1        = 'HGF_low40_250_srl2_hgfv5_1';
options.stats.design_da_hgfv5_1     = 'HGF_low40_250_da_srl2_hgfv5_1';
options.stats.design_da_hgfv5_1_only_CPE = 'HGF_low40_250_da_srl2_hgfv5_1_only_CPE';
% switch options.stats.design
%     case 'epsilon'
%         options.stats.regressors = {'epsi2', 'epsi3'};
% end
options.stats.pValueMode    = 'clusterFWE';
options.stats.exampleID     = '0001';

%-- ERPh ----------------------------------------------------------------%
options.erph.subjectIDs    = options.subjectIDs;
options.erph.overwrite     = 0; % whether to overwrite any previous stats
end

function optionsOut = prssi_get_part_specific_options(optionsIn)

optionsOut = optionsIn;

switch optionsOut.part
    case 'srl_long'
        % subjects
%         optionsOut.tests.subjectIDs    = {'9997', '9996', '6666'};
%         optionsOut.pilots.subjectIDs   = {'0001', '0003', '0005', ...
%             '0012', '0014', '0008'};
        optionsOut.subjectIDs          = {'0303', '0304', '0305', '0306', ...
            '0307', '0309', '0310', '0311', ...
            '0312', '0313', '0314', '0315', ...
            '0316', '0319', '0320', '0321', ...
            '0322', '0323', '0324', '0325', ...
            '0326', '0327', '0328', '0329', ...
            '0330', '0332', '0333', '0335', ...
            '0336', '0337', '0338', '0339', ...
            '0340', '0341', '0342', '0343', ...
            '0344'};
        optionsOut.all.subjectIDs          = {'0303', '0304', '0305', '0306', ...
            '0307', '0309', '0310', '0311', ...
            '0312', '0313', '0314', '0315', ...
            '0316', '0319', '0320', '0321', ...
            '0322', '0323', '0324', '0325', ...
            '0326', '0327', '0328', '0329', ...
            '0330', '0332', '0333', '0335', ...
            '0336', '0337', '0338', '0339', ...
            '0340', '0341', '0342', '0343', ...
            '0344'};
%         optionsOut.noBehav.subjectIDs  = {'0008', '0027', '0095'};
        optionsOut.diffEEG.subjectIDs    = {'0301', '0302'};
        % 0094: no headphones
        
    case 'srl_short'
        % subjects
        optionsOut.subjectIDs          = {'0350', '0351', '0352', '0353', ...
            '0354', '0355', '0356', '0357', ...
            '0358', '0359', '0360', '0361', ...
            '0362', '0363', '0364', '0365', ...
            '0366', '0367', '0368', '0369', ...
            '0370', '0371', '0372', '0373', ...
            '0374', '0375', '0376', '0377', ...
            '0378', '0379', '0380', '0381', ...
            '0382', '0383', '0384', '0385', ...
            '0386', '0387', '0388', '0389'};
        
        optionsOut.all.subjectIDs      = {'0350', '0351', '0352', '0353', ...
            '0354', '0355', '0356', '0357', ...
            '0358', '0359', '0360', '0361', ...
            '0362', '0363', '0364', '0365', ...
            '0366', '0367', '0368', '0369', ...
            '0370', '0371', '0372', '0373', ...
            '0374', '0375', '0376', '0377', ...
            '0378', '0379', '0380', '0381', ...
            '0382', '0383', '0384', '0385', ...
            '0386', '0387', '0388', '0389'};
        optionsOut.problems.subjectIDs = {};
        % 0111: had to repeat MMN
        % 0126: problems with elecs
        
end

optionsOut.srl_long.subjectIDs = {'0303', '0304', '0305', '0306', ...
            '0307', '0309', '0310', '0311', ...
            '0312', '0313', '0314', '0315', ...
            '0316', '0319', '0320', '0321', ...
            '0322', '0323', '0324', '0325', ...
            '0326', '0327', '0328', '0329', ...
            '0330', '0332', '0333', '0335', ...
            '0336', '0337', '0338', '0339', ...
            '0340', '0341', '0342', '0343', ...
            '0344'};
optionsOut.srl_short.subjectIDs = {'0350', '0351', '0352', '0353', ...
    '0354', '0355', '0356', '0357', ...
    '0358', '0359', '0360', '0361', ...
    '0362', '0363', '0364', '0365', ...
    '0366', '0367', '0368', '0369', ...
    '0370', '0371', '0372', '0373', ...
    '0374', '0375', '0376', '0377', ...
    '0378', '0379', '0380', '0381', ...
    '0382', '0383', '0384', '0385', ...
    '0386', '0387', '0388', '0389'};
end
