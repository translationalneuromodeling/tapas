function [fh, MyColors] = physio_get_default_fig_params(xscale, yscale)
% set and return General settings for plots
%
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: physio_get_default_fig_params.m 180 2013-04-23 12:49:11Z kasperla $
%
switch nargin
    case 0
        xscale = 0.5;
        yscale = 0.5;
    case 1
        yscale = 0.5;
end

scrsz = get(0,'ScreenSize');

scrsz = min([1 1 1440 900], scrsz);
fh = figure('Position',[scrsz(1:2) xscale*scrsz(3) yscale*scrsz(4)]);
set(fh, 'WindowStyle', 'docked');
%fh = figure('Position',[scrsz(1:2) xscale*scrsz(3) yscale*scrsz(4)], 'Hidden', 'on');

MyColors = [ ...
    1.0000,         0,         0; ...
    0,    0.5000,         0; ...
    0,         0,         1; ...
    0.7500,         0,    0.7500; ...
    0.7500,    0.7500,         0; ...
    0.2500,    0.2500,    0.2500 ];
set(0,'DefaultAxesColorOrder',MyColors);

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