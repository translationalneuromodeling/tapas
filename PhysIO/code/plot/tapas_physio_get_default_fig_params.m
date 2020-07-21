function [fh, prop, MyColors] = tapas_physio_get_default_fig_params(...
    verbose, convfac, xscale, yscale)
% set and return General settings for plots
%
%  IN
%       visibility  'on'/'off'; sets figure visibility when created
%       convfac     conversion factor (1...Inf) to scale size of text and
%                   lines in plot
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 1
    % If verbose structure is not passed as argument, assume default: figure visibility = 'on'
     visibility = 'on';
else
    % Set default for verbose.show_figs if it is empty or if the field does not exist.
    % Default = true (i.e. show figures)
    if ~isfield(verbose, 'show_figs') || isempty(verbose.show_figs)
        verbose.show_figs = true;
    end
    % Create figure with correct visibility according to show_figs
    if verbose.show_figs
        visibility = 'on';
    else
        visibility = 'off';
    end
end

if nargin < 2    
    convfac = 2; % conversion factor for figure scaling; for laptop display
    % convfac = 4; % for prints in paper
end

if nargin < 3
    xscale = 0.5;
end

if nargin < 4
    yscale = 0.5;
end

scrsz = get(0,'ScreenSize');

scrsz = min([1 1 1440 900], scrsz);
fh = figure('Position',[scrsz(1:2) xscale*scrsz(3) yscale*scrsz(4)], 'visible', visibility);
try
    set(fh, 'WindowStyle', 'docked');
catch err
    if ~strcmp(err.identifier, 'MATLAB:Figure:IncorrectWindowStyle')
        throw(err)
    end
end
%fh = figure('Position',[scrsz(1:2) xscale*scrsz(3) yscale*scrsz(4)], 'Hidden', 'on');

MyColors = [ ...
    1.0000,         0,         0; ...
    0,    0.5000,         0; ...
    0,         0,         1; ...
    0.7500,         0,    0.7500; ...
    0.7500,    0.7500,         0; ...
    0.2500,    0.2500,    0.2500 ];
set(fh,'DefaultAxesColorOrder',MyColors);


%% for printing output
out_width = 21*xscale;
out_height = 30*yscale;
FontName = 'Helvetica';
FontSizeText = 8;
FontSizeAxes = 8;
LineWidth = 3;
set(gcf,'DefaultLineLineWidth', LineWidth*convfac/2);
set(gcf,'DefaultAxesLineWidth', LineWidth*convfac/2);
set(gcf,'DefaultAxesFontName', FontName);
set(gcf,'DefaultAxesFontSize', FontSizeAxes*convfac);
set(gcf, 'DefaultTextFontSize', FontSizeText*convfac);
set(gcf, 'PaperUnits', 'centimeter');
set(gcf, 'PaperPosition', [0 0 out_width out_height]*convfac);
set(gcf, 'PaperSize', [out_width out_height]*convfac);

prop.FontName = FontName;
prop.FontSizeText = FontSizeText*convfac;
prop.FontSizeAxes = FontSizeAxes*convfac;
prop.LineWidth = LineWidth*convfac;
prop.colors = MyColors;

%         %set color order for plot: ECG should be red, because blood :-)
%     MyColors = [ ...
%         1.0000,         0,         0; ...
%         0,         0,         1; ...
%         0,    0.5000,         0; ...
%         0.7500,         0,    0.7500; ...
%         0.7500,    0.7500,         0; ...
%         0.2500,    0.2500,    0.2500 ];
%     set(0,'DefaultAxesColorOrder',MyColors);


end
