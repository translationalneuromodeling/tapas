function [fh yMin, yMax] = tapas_uniqc_plot_abs_image(Y,iDynSli,fh, yMin, yMax, ...
    colorMap, colorBar)
%simple plotting routine for one dynamic within a 3D-dataset
%
%   [fh yMin, yMax] = tapas_uniqc_plot_image_diagnostics(Y,iDynSli,fh, yMin, yMax)
%
% IN
%   Y           [nVoxelX nVoxelY, nSlices, nVolumes] real valued data matrix
%               OR
%               [nSamples, nCoils, nDynSlis] real-valued data matrix
%   iDynSli     index of which dynamic/slice shall be plotted
%   fh          figure handle to be plotted into; if empty or missing, new figure is
%               created
%   yMin        [nPlots,1] min value in ylim, one for each set of coils, if
%               empty or missing, newly created from data limits
%   yMax        [nPlots,1] max value in ylim, one for each set of coils, if
%               empty or missing, newly created from data limits
%
% OUT
%   fh          figure handle plotted into
%   yMin        [nPlots,2] min value in ylim, one for each set of coils,
%               abs and phase
%   yMax        [nPlots,2] max value in ylim, one for each set of coils
%               abs and phase
% EXAMPLE
%   [fh yMin, yMax] = tapas_uniqc_plot_image_diagnostics(Y,iDynSli,fh, yMin, yMax, coilPlots)
%
%   See also plotTrajDiagnostics guiTrajDiagnostics

% Author: Lars Kasper
% Created: 2013-01-06
% Copyright 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.



if nargin < 3 || isempty(fh)
    fh = figure('Name','Video of ImageDiagnostics', 'WindowStyle', 'normal');
else
    figure(fh);
end

% determine plot limits
if nargin < 5 || isempty(yMin)
    validIndices = find(~isinf(Y) & ~isnan(Y));
    yMin = min(Y(validIndices));
    yMax = max(Y(validIndices));
end

if nargin < 6
    colorMap = 'gray';
end

if nargin < 7
    colorBar = 'off';
end

doPlotColorBar = strcmpi(colorBar, 'on');

% plot abs data always
imagesc(Y(:,:,iDynSli));
%colormap gray; axis image;
colormap(colorMap); axis image;

caxis([yMin, yMax]);

if doPlotColorBar
    colorbar;
end

stringTitle = sprintf('abs, iDynSli = %d', iDynSli);
if exist('suptitle', 'builtin')
    suptitle(stringTitle);
else
    title(stringTitle);
end

end
