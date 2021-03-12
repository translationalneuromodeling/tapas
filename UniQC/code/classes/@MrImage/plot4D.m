function fh = plot4D(this, varargin)
%plots an MR image per slice
%
%   Y  = MrImage
%   fh = Y.plot4D('ParameterName', ParameterValue)
%
% IN
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%               'plotType'          Type of plot that is created
%                                       'montage'   images are plotted as
%                                                   montages of slices or
%                                                   volumes
%                                       'labeledMontage'
%                                                   as montage, but with
%                                                   slice/volume labels
%                                                   (default)
%                                       'overlay'   overlays of images are
%                                                   plotted with different
%                                                   colormaps (e.g. for
%                                                   activation maps, mask
%                                                   visualization)
%                                       'spm'       uses display functions
%                                                   from SPM (spm_display/
%                                                   spm_check_registration)
%                                                   to visualize 3D volumes
%                                                   with header information
%                                                   applied ("world space")
%                                                   Note: if multiple
%                                                   selected volumes are
%                                                   specified,
%                                                   spm_check_registration
%                                                   is used
%                                       'spmInteractive' /'spmi'
%                                                   same as SPM, but keeps
%                                                   temporary nifti files to
%                                                   allow clicking into spm
%                                                   figure
%                                       '3D'/'3d'/'ortho'
%                                                   See also tapas_uniqc_view3d plot3
%                                                   Plots 3 orthogonal
%                                                   sections
%                                                   (with CrossHair) of
%                                                   3D image interactively
%
%               'displayRange'      [1,2] vector for pixel value = black and
%                                                    pixel value = white
%               'signalPart'        for complex data, defines which signal
%                                   part shall be extracted for plotting
%                                       'all'       - take signal as is
%                                                     (default for
%                                                     real-valued data)
%                                       'abs'       - absolute value
%                                                     (default for complex
%                                                     data)
%                                       'phase'     - phase of signal
%                                       'real'      - real part of signal
%                                       'imag'      - imaginary part of
%                                                     signal
%               'plotMode'          transformation of data before plotting
%                                   'linear' (default), 'log'
%               'selectedVolumes'   [1,nVols] vector of selected volumes to
%                                             be displayed
%               'selectedSlices'    [1,nSlices] vector of selected slices to
%                                               be displayed
%                                   choose Inf to display all volumes
%               'sliceDimension'    (default: 3) determines which dimension
%                                   shall be plotted as a slice
%               'rotate90'          default: 0; 0,1,2,3; rotates image
%                                   by multiple of 90 degrees AFTER
%                                   flipping slice dimensions
%               'useSlider'         true or false
%                                   provides interactive slider for
%                                   slices/volumes;
%                                   assumes default:    selectedSlices = Inf
%                                                       selectedVolumes = Inf
%               'fixedWithinFigure' determines what dimension is plotted in
%                                   (subplots of) 1 figure
%                                   'slice(s)'    all slices in 1 figure;
%                                   new figure for each volume
%                                   'volume(s)'   all volumes in 1 figurel
%                                   new figure for each slice
%               'colorMap'          string, any matlab colormap name
%                                   e.g. 'jet', 'gray'
%               'colorBar',         'on' or 'off' (default)
%                                   where applicable, determines whether
%                                   colorbar with displayRange shall be plotted
%                                   in figure;
%               'overlayImages'     (cell of) MrImages that will be
%                                   overlayed
%               'overlayMode'       'edge', 'mask', 'map'
%                                   'edge'  only edges of overlay are
%                                           displayed
%                                   'mask'  every non-zero voxel is
%                                           displayed (different colors for
%                                           different integer values, i.e.
%                                           clusters'
%                                   'map'   thresholded map in one colormap
%                                           is displayed (e.g. spmF/T-maps)
%                                           thresholds from
%                                           overlayThreshold
%               'overlayThreshold'  [minimumThreshold, maximumThreshold]
%                                   tresholds for overlayMode 'map'
%                                   default: [-Inf, Inf] = [minValue, maxValue]
%                                   everything below minValue will not be
%                                   displayed;
%                                   everything above maxValue
%                                   will have brightest color
%               'overlayAlpha'      transparency value of overlays
%                                   (0 = transparent; 1 = opaque; default: 0.1)
%
%
% OUT
%   fh          [nFigures,1] vector of figure handles
%
% EXAMPLE
%
%   Y.plot4D('selectedVolumes', [6:10])
%   Y.plot4D('displayRange', [0 1000])
%   Y.plot4D('useSlider', true, 'selectedVolumes', Inf);
%
%   See also

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-05-21
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% for complex data, plot absolute value per default
if isreal(this)
    defaults.signalPart         = 'all';
else
    defaults.signalPart         = 'abs';
end

defaults.plotType               = 'labeledMontage';
defaults.plotMode               = 'linear';
defaults.selectedVolumes        = 1;
defaults.selectedX              = Inf;
defaults.selectedY              = Inf;
defaults.selectedSlices         = Inf;
defaults.sliceDimension         = 3;
defaults.rotate90               = 0;
defaults.displayRange           = [];
defaults.useSlider              = false;
defaults.fixedWithinFigure      = 'volume';
defaults.colorMap               = 'gray';
defaults.colorBar               = 'off';
defaults.overlayImages          = {};
defaults.overlayMode            = 'mask';
defaults.overlayThreshold       = [];
defaults.overlayAlpha           = 0.1;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

doPlotColorBar = strcmpi(colorBar, 'on');
doPlotOverlays = any(strcmpi(plotType, {'overlay', 'overlays'})) || ...
    ~isempty(overlayImages);


% slider enables output of all Slices and Volumes per default, strip data
% again under this assumption, if slider is used for display
if useSlider
    defaults.selectedVolumes = Inf;
    defaults.selectedSlices = Inf;
    args = tapas_uniqc_propval(varargin, defaults);
    tapas_uniqc_strip_fields(args);
end

% Assemble parameters for data extraction into one structure
argsExtract = struct('sliceDimension', sliceDimension, ...
    'selectedX', selectedX, 'selectedY', selectedY, ...
    'selectedSlices', selectedSlices, 'selectedVolumes', selectedVolumes, ...
    'plotMode', plotMode, 'rotate90', rotate90, 'signalPart', signalPart);


if isempty(this.data)
    error('tapas:uniqc:MrImage:plot4D:EmptyDataMatrix', ...
        'Data matrix empty for MrImage-object %s', this.name);
end


% retrieve plot data without actually plotting...
if doPlotOverlays
    argsOverlays = struct('sliceDimension', sliceDimension, ...
        'selectedSlices', selectedSlices, 'selectedVolumes', selectedVolumes, ...
        'plotMode', plotMode, 'rotate90', rotate90, 'signalPart', signalPart, ...
        'overlayMode', overlayMode, 'overlayThreshold',  overlayThreshold, ...
        'doPlot', true);
    
    [fh, dataPlot] = this.plot_overlays(overlayImages, argsOverlays);
    return
else
    if isempty(displayRange)
        [dataPlot, displayRange, resolution_mm] = this.extract_plot4D_data(argsExtract);
    else
        [dataPlot, ~, resolution_mm] = this.extract_plot4D_data(argsExtract);
    end
end


nVolumes = size(dataPlot,4);
nSlices = size(dataPlot,3);

if isinf(selectedSlices)
    selectedSlices = 1:nSlices;
end

if isinf(selectedVolumes)
    selectedVolumes = 1:nVolumes;
end

% slider view
if useSlider
    % useSlider is not a plotType, since it shall be combined with all
    % plot-types (overlays, montages) in a later version of this code
    
    % tapas_uniqc_slider4d(dataPlot, @(varargin) ...
    %      tapas_uniqc_plot_abs_image(varargin{:}, colorMap), ...
    %     nSlices);
    
    
    tapas_uniqc_slider4d(dataPlot, @(Y,iDynSli, fh, yMin, yMax) ...
        tapas_uniqc_plot_abs_image(Y,iDynSli, fh, yMin, yMax, colorMap, colorBar), ...
        nSlices, displayRange(1), displayRange(2), this.name);
    
    % to also plot phase:
    %    tapas_uniqc_slider4d(dataPlot, @tapas_uniqc_plot_image_diagnostics, ...
    %        nSlices);
    
else
    switch lower(plotType)
        case {'3d', 'ortho'}
            this.plot3d(argsExtract);
        case {'spm', 'spminteractive', 'spmi'}
            % calls spm_image-function (for single volume) or
            % spm_check_registration (multiple volumes)
            
            % get current filename, make sure it is nifti-format
            fileName = this.parameters.save.fileName;
            fileNameNifti = fullfile(this.parameters.save.path, ...
                regexprep(fileName, '\..*$', '\.nii'));
            
            % create nifti file, if not existing
            % TODO: how about saved objects with other file names
            if ~exist(fileNameNifti, 'file')
                this.save(fileNameNifti);
            end
            
            % select Volumes
            fileNameVolArray = tapas_uniqc_get_vol_filenames(fileNameNifti);
            
            % display image
            if numel(selectedVolumes) > 1
                spm_check_registration( ...
                    fileNameVolArray{selectedVolumes});
            else
                spm_image('Display', fileNameVolArray{selectedVolumes});
            end
            
            % delete temporary files for display
            if strcmpi(this.parameters.save.keepCreatedFiles, 'none')
                
                switch lower(plotType)
                    case {'spminteractive', 'spmi'}
                        input('Press Enter to leave interactive mode');
                end
                
                delete(fileNameNifti);
                [stat, mess, id] = rmdir(this.parameters.save.path);
            end
            
            % restore original file name
            this.parameters.save.fileName = fileName;
            
            
        case {'montage', 'labeledmontage'}
            
            if strcmpi(plotType, 'labeledMontage')
                
                stringLabelSlices = cellfun(@(x) num2str(x), ...
                    num2cell(selectedSlices), 'UniformOutput', false);
                stringLabelVolumes = cellfun(@(x) num2str(x), ...
                    num2cell(selectedVolumes), 'UniformOutput', false);
            else
                stringLabelSlices = [];
                stringLabelVolumes = [];
            end
            
            switch lower(fixedWithinFigure);
                case {'volume', 'volumes'}
                    for iVol = 1:nVolumes
                        
                        nFrames = size(dataPlot, 3);
                        
                        stringTitle = sprintf('%s - volume %d', this.name, ...
                            selectedVolumes(iVol));
                        fh(iVol,1) = figure('Name', stringTitle, 'WindowStyle', 'docked');
                        
                        tapas_uniqc_labeled_montage(permute(dataPlot(:,:,:,iVol), [1, 2, 4, 3]), ...
                            'DisplayRange', displayRange, ...
                            'LabelsIndices', stringLabelSlices);
                        
                        set(gca, 'DataAspectRatio', abs([resolution_mm(1) resolution_mm(2), 1]));
                        
                        title(tapas_uniqc_str2label(stringTitle));
                        if doPlotColorBar
                            colorbar;
                        end
                        colormap(colorMap);
                        
                    end
                    
                case {'slice', 'slices'}
                    
                    for iSlice = 1:nSlices
                        stringTitle = sprintf('%s - slice %d', this.name, ...
                            selectedSlices(iSlice));
                        fh(iSlice,1) = figure('Name', stringTitle, 'WindowStyle', 'docked');
                        tapas_uniqc_labeled_montage(dataPlot(:,:,iSlice,:), ...
                            'DisplayRange', displayRange, ...
                            'LabelsIndices', stringLabelVolumes);
                        
                        set(gca, 'DataAspectRatio', [resolution_mm(1) resolution_mm(2), 1]);
                        title(tapas_uniqc_str2label(stringTitle));
                        if doPlotColorBar
                            colorbar;
                        end
                        colormap(colorMap);
                    end
                    
            end % fixedWithinFigure
    end % plotType
end % use Slider
end