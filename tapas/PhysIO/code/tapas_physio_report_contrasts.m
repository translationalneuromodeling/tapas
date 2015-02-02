function args = tapas_physio_report_contrasts(varargin)
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
%                   'fileSpm', 'analysisFolder/SPM.mat'
%                   'fileStructural', 'mean.nii', ...
%                   'indReportPhysContrasts', [1:5] ...
%                   )
%
% IN
%   
%   Required parameters:
%
%                  fileReport: post-script file to print results to
%              fileStructural: structural underlay for results,
%                              e.g. 'mean.nii'
%                     fileSpm: SPM.mat holding physiological regressors, 
%                              e.g.'SPM.mat'
%
%   Optional Parameters:
%
%                  pathPhysIO:  path of physIO Toolbox code
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
%
% OUT
%   args    structure of default and updated arguments used in this
%           function; the fields of args hold all possible options of this
%           function
%
% EXAMPLE
%   tapas_physio_report_contrasts
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-10-16
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 464 2014-04-27 11:58:09Z kasperla $

%% START #MOD =============================================================

% general paths study
defaults.titleGraphicsWindow = '';
% PhysIO Toolbox code should be in same folder as this file
defaults.pathPhysIO      = fileparts(mfilename('fullpath'));
defaults.fileReport      = 'physio_report_contrasts.ps'; % where contrast maps are saved
defaults.fileStructural  = 'mean.nii';
defaults.fileSpm         = 'SPM.mat';
defaults.drawCrosshair   = true;

% names of physiological contrasts to be reported
% namesPhysContrasts = {
%             'All Phys Regressors'
%             'Cardiac Regressors'
%             'Respiratory Regressors'
%             'Cardiac X Respiratory Interaction'
%             'Movement Regressors'
%             };

defaults.namesPhysContrasts = {
    'All Phys'
    'Cardiac'
    'Respiratory'
    'Card X Resp Interation'
    'HeartRateVariability'
    'RespiratoryVolumePerTime'
    'Movement'
    };

% selection of physiological contrasts to be reported, corresponding to
% namesPhysContrasts order
defaults.indReportPhysContrasts = 2;

defaults.reportContrastThreshold     = 0.001; % 0.05; 0.001;
defaults.reportContrastCorrection    = 'none'; % 'FWE'; 'none';
%reportContrastPosition      = [0 -15 -2*16]; 'max'; % 'max' to jump to max; or [x,y,z] in mm
%fovMillimeter               = 50; %mm; choose 0 to plot whole FOV (bounding box)
defaults.reportContrastPosition      = 'max'; % 'max' to jump to max; or [x,y,z] in mm
defaults.fovMillimeter               = 0; %mm; choose 0 to plot whole FOV (bounding box)

% if true, voxel space (parallel to slices), not world space (with interpolation) is used
defaults.doPlotSliceParallel          = true; 

physio                                = tapas_physio_new('RETROICOR');
defaults.model                        = physio.model; % holding number of physiological regressors

% END #MOD
%% ========================================================================

args = propval(varargin, defaults);
tapas_physio_strip_fields(args);

load(fileSpm);
nContrasts = numel(indReportPhysContrasts);

% Temporarily set window style to undocked, so that SPM opens as usual
tmpWindowStyle = get(0, 'DefaultFigureWindowStyle');
set(0, 'DefaultFigureWindowStyle', 'normal');

%% Check whether contrasts already exist in SPM.mat
indContrasts = zeros(nContrasts,1);
for c = 1:nContrasts
    iC = indReportPhysContrasts(c);
    indContrasts(c) = tapas_physio_check_get_xcon_index(SPM, ...
        namesPhysContrasts{iC});
end


%% Generate contrasts only if not already existing

if ~isempty(model)
    
    matlabbatch = tapas_physio_check_prepare_job_contrasts(fileSpm, ...
        model, SPM, indReportPhysContrasts, pathPhysIO, namesPhysContrasts);
    matlabbatch{1}.spm.stats.con.consess(find(indContrasts)) = [];
    if ~isempty(matlabbatch{1}.spm.stats.con.consess)
        spm_jobman('run', matlabbatch);
        load(fileSpm);
    end
    
end

%% report contrasts
for c = 1:nContrasts
    iC = indReportPhysContrasts(c);
    indContrasts(c) = tapas_physio_check_get_xcon_index(SPM, ...
        namesPhysContrasts{iC});
    load(fullfile(pathPhysIO, 'tapas_physio_check_job_report'));
    matlabbatch{1}.spm.stats.results.spmmat = cellstr(fileSpm);
    matlabbatch{1}.spm.stats.results.conspec.titlestr = [titleGraphicsWindow ' - ' namesPhysContrasts{iC}];
    matlabbatch{1}.spm.stats.results.conspec.contrasts = indContrasts(c);
    
    % contrast report correction
    matlabbatch{1}.spm.stats.results.conspec.thresh = reportContrastThreshold;
    matlabbatch{1}.spm.stats.results.conspec.threshdesc = reportContrastCorrection;
    
    spm_jobman('run', matlabbatch);                     % report result
    %                     spm_print(fileReport)
    xSPM = evalin('base', 'xSPM');
    hReg = evalin('base', 'hReg');
    
    spm_sections(xSPM,hReg, fileStructural);                % overlay structural
    
    % voxel, not world space
    if doPlotSliceParallel
        spm_orthviews('Space',1)
    end
    
    spm_orthviews('Zoom', fovMillimeter); % zoom to FOV*2 view
    spm_orthviews('Interp', 0); % next neighbour interpolation plot
 
    if isequal(reportContrastPosition, 'max');
        spm_mip_ui('Jump',spm_mip_ui('FindMIPax'),'glmax'); % goto global max
    else
        spm_mip_ui('SetCoords', reportContrastPosition, ...
            spm_mip_ui('FindMIPax')); % goto global max
    end
    
    % to be able to turn off the blue Crosshir
    if ~ drawCrosshair
        spm_orthviews('Xhairs','off');

    end
    
    %         spm_orthviews - spm.st.blobs.cbar


    % spm_print always prepend current directory to print-file
    % name :-(
    [pathReport, filenameReport] = fileparts(fileReport);
    if ~isempty(pathReport)
        pathTmp = pwd;
        cd(pathReport);
        spm_print(filenameReport);
        cd(pathTmp);
    end
end

titstr = [titleGraphicsWindow, ' - SPM.xX.X'];
title(regexprep(titstr,'_','\\_'));
set(gcf,'Name', titstr);

set(0, 'DefaultFigureWindowStyle', tmpWindowStyle);

