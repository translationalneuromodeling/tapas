function varargout = tapas_physio_report_contrasts(varargin)
% This function reports all relevant F-contrast-maps for physIO-created regressors
% in a given GLM
%
%  - Section plots on an anatomical overlay are created and saved as
%    post-script file with one page for each contrasts
%
%  - Input parameters are specified via name/value pairs, e.g.
%
%
%   args = tapas_physio_report_contrasts(...
%                   'fileReport', 'physio.ps', ...
%                   'fileSpm', 'analysisFolder/SPM.mat', ...
%                   'filePhysIO', 'analysisFolder/physio.mat', ...
%                   'fileStructural', 'anatomyFolder/warpedAnatomy.nii')
%
% IN
%
%   Required parameters:
%
%                  fileReport: sets the path and file name for the postscript 
%                              file your contrast report is printed to. 
%                              A reasonable (and default) location is the 
%                              single subject analysis folder so the file 
%                              resides with the GLM (fileSPM)
%                              default: 'pathSpm/physio_report_contrasts.ps';
%              fileStructural: structural underlay for results,
%                              e.g. 'mean.nii'
%                     fileSpm: SPM.mat holding physiological regressors,
%                              e.g.'SPM.mat'
%                  filePhysIO:  mat-file where PhysIO-object was saved
%                               e.g. physio.mat
%
%   Optional Parameters:
%
%          namesPhysContrasts: cell Array of contrast names in design matrix
%                              e.g. {'All Phys', 'Cardiac', 'Respiratory',
%                               'Card X Resp Interation',
%                               'HeartRateVariability',
%                               'RespiratoryVolumePerTime', 'Movement'}
%
%      indReportPhysContrasts: vector of contrast indices to be reported
%                               e.g. [1:7] for all physiological contrasts
%     reportContrastThreshold: 0.001
%    reportContrastCorrection: 'none' or 'FWE'
%    reportContrastMax:        maximum value of contrast colormap
%                              to scale different contrasts with equal
%                              F-values (default: Inf, scales to max F of
%                              map)
%      reportContrastPosition: 'max' or [1,3] vector of crosshair position
%                               (in mm)
%               fovMillimeter: field of view in mm; set to 0 for full FOV
%         doPlotSliceParallel: if true, slices are plotted parallel to
%                              their slice acquisition direction
%                       model: physio.model-substructure holding
%                              model orders, i.e. .c .r .cr
%                              See also tapas_physio_new
%         titleGraphicsWindow: additional title prepended to contrast name
%                              in each plot
%       saveTable              true of false (default)
%                              prints screenshot of results table to
%                              ps-file as well
%
% OUT
%   args    structure of default and updated arguments used in this
%           function; the fields of args hold all possible options of this
%           function
%
% EXAMPLE
%   tapas_physio_report_contrasts
%
%   See also tapas_physio_overlay_contrasts

% Author: Lars Kasper
% Created: 2014-10-16
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



%% START #MOD =============================================================

% general paths study
defaults.titleGraphicsWindow = '';
% PhysIO Toolbox code should be in same folder as this file
defaults.filePhysIO      = 'physio.mat';
defaults.fileReport      = 'physio_report_contrasts.ps'; % where contrast maps are saved
defaults.fileStructural  = 'mean.nii';
defaults.fileSpm         = 'SPM.mat';
defaults.drawCrosshair   = true;

defaults.namesPhysContrasts = tapas_physio_get_contrast_names_default();

% selection of physiological contrasts to be reported, corresponding to
% namesPhysContrasts order
defaults.indReportPhysContrasts = 1:9;

defaults.reportContrastThreshold     = 0.001; % 0.05; 0.001;
defaults.reportContrastCorrection    = 'none'; % 'FWE'; 'none';
defaults.reportContrastMax           = Inf;
defaults.reportContrastPosition      = 'max'; % 'max' to jump to max; or [x,y,z] in mm
defaults.fovMillimeter               = 0; %mm; choose 0 to plot whole FOV (bounding box)

% if true, voxel space (parallel to slices), not world space (with interpolation) is used
defaults.doPlotSliceParallel          = true;
defaults.saveTable                    = false;
physio                                = tapas_physio_new('RETROICOR');
defaults.model                        = physio.model; % holding number of physiological regressors

% END #MOD
%% ========================================================================

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);

spm('defaults', 'FMRI');

% make sure to use absolute paths from now on...
fileSpm = tapas_physio_filename2path(fileSpm);
fileStructural = tapas_physio_filename2path(fileStructural);
fileReport = tapas_physio_filename2path(fileReport);
filePhysIO = tapas_physio_filename2path(filePhysIO);

load(fileSpm);
nContrasts = numel(indReportPhysContrasts);

if ~exist(fileStructural, 'file')
    % take standard structural from SPM, if overlay file does not exist
    pathSpm = fileparts(which('spm'));
    fileStructural = fullfile(pathSpm, 'canonical', 'avg152T1.nii');
end

% if input file given, load PhysIO-object
if exist(filePhysIO, 'file')
    load(filePhysIO, 'physio');
    model = physio.model;
end

% Temporarily set window style to undocked, so that SPM opens as usual
tmpWindowStyle = get(0, 'DefaultFigureWindowStyle');
set(0, 'DefaultFigureWindowStyle', 'normal');

% create physiological contrasts that don't exist so far and can be created
% due to the model components included
SPM = tapas_physio_create_missing_physio_contrasts(SPM, model, namesPhysContrasts);

%% report contrasts with overlays

for c = 1:nContrasts
    iC = indReportPhysContrasts(c);
    idxContrasts(c) = tapas_physio_check_get_xcon_index(SPM, ...
        namesPhysContrasts{iC});
end

idxContrasts = idxContrasts(idxContrasts > 0);

% TODO: harmonize option labels in both functions
tapas_physio_overlay_contrasts( ...
    'idxContrasts', idxContrasts, ...
    'titleGraphicsWindow', titleGraphicsWindow, ...
    'fileReport', fileReport, ...
    'fileSpm', fileSpm, ...
    'fileStructural', fileStructural, ...
    'threshold', reportContrastThreshold, ...
    'correction', reportContrastCorrection, ...
    'doPlotSliceParallel', doPlotSliceParallel, ...
    'fovMillimeter', fovMillimeter, ...
    'position', reportContrastPosition, ...
    'drawCrosshair', drawCrosshair, ...
    'colorbarMax', reportContrastMax, ...
    'saveTable', saveTable);

if nargout
    varargout{1} = args;
end
