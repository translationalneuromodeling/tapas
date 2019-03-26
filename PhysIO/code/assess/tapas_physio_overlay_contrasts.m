function varargout = tapas_physio_overlay_contrasts(varargin)
% as Contrast Results report, but overlaying activations on brain image 
% instead of showing results table
%
%  - Section plots on an anatomical overlay are created and saved as
%    post-script file with one page for each contrast
%
%  - Input parameters are specified via name/value pairs, e.g.
%
%   fileReport = tapas_physio_overlay_contrasts(...
%                   'fileReport', 'contrast_report.ps', ...
%                   'fileSpm', 'analysisFolder/SPM.mat', ...
%                   'fileStructural', 'anatomyFolder/warpedAnatomy.nii')
%
% IN
%
%   Required parameters:
%
%                  fileReport: post-script file to print results to
%              fileStructural: structural underlay for results,
%                              e.g. 'mean.nii'
%                               default: spm/canonical/avg152T1.nii
%                     fileSpm: SPM.mat holding physiological regressors,
%                              e.g.'SPM.mat'
%
%   Optional Parameters:
%
%
%       idxContrasts:           vector of contrast indices to be reported
%                               e.g. [1:7] for all physiological contrasts
%       threshold:                0.001
%       correction:             for multiple comparisons: '
%                               none' (default) or 'FWE'
%       colorbarMax:            maximum value of contrast colormap 
%                               to scale different contrasts with equal
%                               F-values (default: Inf, scales to max F of
%                               map)
%       drawCrosshair           true (default) or false.
%       position:                'max' or [1,3] vector of crosshair position
%                               (in mm)
%       fovMillimeter:          field of view in mm; set to 0 for full FOV
%       doPlotSliceParallel:    if true, slices are plotted parallel to
%                               their slice acquisition direction
%       titleGraphicsWindow:    additional title prepended to contrast name
%                               in each plot
%       saveTable               true of false (default)
%                               prints screenshot of results table to
%                               ps-file as well
%
% OUT
%   args    structure of default and updated arguments used in this
%           function; the fields of args hold all possible options of this
%           function
%
%   See also tapas_physio_report_contrasts

% Author:   Lars Kasper
% Created:  2017-01-05
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the Zurich fMRI Methods Evaluation Repository, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.




%% START #MOD =============================================================

% general paths study
defaults.titleGraphicsWindow    = '';
% PhysIO Toolbox code should be in same folder as this file
defaults.fileReport             = 'report_contrasts.ps'; % where contrast maps are saved
defaults.fileStructural         = 'mean.nii';
defaults.fileSpm                = 'SPM.mat';
defaults.drawCrosshair          = true;

defaults.idxContrasts           = Inf;

defaults.threshold              = 0.001; % 0.05; 0.001;
defaults.correction             = 'none'; % 'FWE'; 'none';
defaults.colorbarMax            = Inf;   
defaults.position               = 'max'; % 'max' to jump to max; or [x,y,z] in mm
defaults.fovMillimeter          = 0; %mm; choose 0 to plot whole FOV (bounding box)
defaults.saveTable              = 0;
% if true, voxel space (parallel to slices), not world space (with interpolation) is used
defaults.doPlotSliceParallel    = true;


% END #MOD
%% ========================================================================

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);

spm('defaults', 'FMRI');

% make sure to use absolute paths from now on...
fileSpm = tapas_physio_filename2path(fileSpm);
fileStructural = tapas_physio_filename2path(fileStructural);
fileReport = tapas_physio_filename2path(fileReport);

load(fileSpm);

if isinf(idxContrasts)
    idxContrasts = 1:numel(SPM.xCon);
end
nContrasts = numel(idxContrasts);

if ~exist(fileStructural, 'file')
    % take standard structural from SPM, if overlay file does not exist
    pathSpm = fileparts(which('spm'));
    fileStructural = fullfile(pathSpm, 'canonical', 'avg152T1.nii');
end

% Temporarily set window style to undocked, so that SPM opens as usual
tmpWindowStyle = get(0, 'DefaultFigureWindowStyle');
set(0, 'DefaultFigureWindowStyle', 'normal');


%% report contrasts
pathBeforeReport = pwd;
for idxContrast = idxContrasts
    
        load('tapas_physio_check_job_report');
        matlabbatch{1}.spm.stats.results.spmmat = cellstr(fileSpm);
        matlabbatch{1}.spm.stats.results.conspec.titlestr = [titleGraphicsWindow ' - ' SPM.xCon(idxContrast).name];
        matlabbatch{1}.spm.stats.results.conspec.contrasts = idxContrast;
        
        % contrast report correction
        matlabbatch{1}.spm.stats.results.conspec.thresh = threshold;
        matlabbatch{1}.spm.stats.results.conspec.threshdesc = correction;
        
        spm_jobman('run', matlabbatch); % report result
        
        % Change directory, ince spm_print always prepend current directory 
        % to print-file name :-(
        [pathReport, filenameReport] = fileparts(fileReport);
        if isempty(pathReport)
            pathReport = pwd;
        end
        pathTmp = pwd;
        
        cd(pathReport);
        
        if saveTable
            spm_print(fileReport)
        end
        
        xSPM = evalin('base', 'xSPM');
        hReg = evalin('base', 'hReg');
        
        spm_sections(xSPM,hReg, fileStructural);  % overlay structural
        
        % voxel, not world space
        if doPlotSliceParallel
            spm_orthviews('Space',1)
        end
        
        spm_orthviews('Zoom', fovMillimeter); % zoom to FOV*2 view
        spm_orthviews('Interp', 0); % next neighbour interpolation plot
        
        if isequal(position, 'max');
            spm_mip_ui('Jump',spm_mip_ui('FindMIPax'),'glmax'); % goto global max
        else
            spm_mip_ui('SetCoords', position, ...
                spm_mip_ui('FindMIPax')); % goto global max
        end
        
        % to be able to turn off the blue Crosshair
        if ~drawCrosshair
            spm_orthviews('Xhairs','off');
        end
        
        % spm_orthviews - spm.st.blobs.cbar, changes colorbar
        if ~isinf(colorbarMax)
            spm_orthviews('SetBlobsMax', 1, 1, colorbarMax)
        end
       
        spm_print(filenameReport);
        cd(pathTmp);
end
cd(pathBeforeReport);

stringTitle = ['Assess: ' titleGraphicsWindow, ' - SPM.xX.X'];
title(regexprep(stringTitle,'_','\\_'));
set(gcf,'Name', stringTitle);

set(0, 'DefaultFigureWindowStyle', tmpWindowStyle);

if nargout
    varargout{1} = args;
end