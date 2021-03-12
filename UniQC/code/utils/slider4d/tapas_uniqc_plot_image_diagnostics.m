function [fh yMin, yMax] = tapas_uniqc_plot_image_diagnostics(Y,iDynSli,fh, yMin, yMax)
%plots abs and phase of specific dyn/slice of complex data, for different
%sets of coils, adjusts plot limits automatically
%
%   [fh yMin, yMax] = tapas_uniqc_plot_image_diagnostics(Y,iDynSli,fh, yMin, yMax)
%
% IN
%   Y           [nVoxelX nVoxelY, nSlices, nVolumes] data matrix
%               OR
%               [nSamples, nCoils, nDynSlis] complex data matrix
%               OR {absY, angY}, i.e. cell(2,1) which matrices of
%               dimension [nSamples, nCoils, nDynSlis] holding magnitude
%               and phase of Y
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


if iscell(Y)
    absY = Y{1};
    angY = Y{2};
else
    absY = abs(Y);
    angY = angle(Y);
end

if nargin < 3 || isempty(fh)
    fh = figure('Name','Video of ImageDiagnostics');
else
    figure(fh);
end

nPlots = 1;
% determine plot limits
if nargin < 5 || isempty(yMin)
    for iPlot = 1:nPlots
        yMin(iPlot, 1) = min(min(min(absY(:,:,:,:))));
        yMax(iPlot, 1) = max(max(max(absY(:,:,:,:))));
        yMin(iPlot, 2) = min(min(min(angY(:,:,:,:))));
        yMax(iPlot, 2) = max(max(max(angY(:,:,:,:))));
    end
end

% replicate number of scalings for all images
if numel(yMin) == 1
    yMin(1,2) = yMin(1,1);
    yMax(1,2) = yMax(1,1);
end


isDataComplex = any(angY(:));
iPlot = 1;


% plot phase data, if existing
if isDataComplex
    nColumns = 2;
    hs(iPlot,nColumns) = subplot(nPlots,nColumns,iPlot+nPlots);
    imagesc(angY(:,:,iDynSli));
    colormap gray; axis image;
    title('phase');
    caxis([yMin(iPlot, 2), yMax(iPlot, 2)]);
else
    nColumns = 1;
end

% plot abs data always
hs(iPlot,1) = subplot(nPlots,nColumns,iPlot);
imagesc(absY(:,:,iDynSli));
colormap gray; axis image;
title('abs');
caxis([yMin(iPlot, 1), yMax(iPlot, 1)]);

stringTitle = sprintf('iDynSli = %d', iDynSli);
if exist('suptitle', 'builtin')
    suptitle(stringTitle);
else
    title(stringTitle);
end

drawnow;
%weird, deletes last legend...
%legend(hs(nPlots,2),cellfun(@num2str, num2cell(coilPlots{iPlot}), 'UniformOutput', false))
%drawnow;
%pause(0.1);
%end
if isDataComplex
    linkaxes(hs(:),'xy');
end
end
