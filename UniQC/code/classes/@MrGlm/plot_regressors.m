function [fh, this] = plot_regressors(this, module)
% Plots regressors
%
%   G = MrGlm()
%   G.plot_regressors(inputs)
%
% This is a method of class MrGlm.
%
% IN
%   module      which type of regressors shall be plotted
%               'realign'   Realignment parameters.
%                           See also MrImage.realign
%               'physio'    Physiological noise regressors.
%                           See also tapas_physio_new()
% OUT
%
% EXAMPLE
%   plot_regressors
%
%   See also MrGlm MrImage MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


switch lower(module)
    case 'realign'
        
        rp = this.regressors.realign;
        
        if ~isempty(rp)
            t = 1:size(rp,1);
            stringTitle = sprintf('Realignment Parameters');
            fh = figure('WindowStyle', 'docked');
            set(fh, 'Name', stringTitle);
            
            hs(1) = subplot(2,1,1);
            plot(t, rp(:,1:3));
            title('Translation Parameters (mm)');
            xlabel('number of scans'); ylabel('translation (mm)');
            legend('x','y', 'z');
            
            hs(2) = subplot(2,1,2);
            plot(t, rp(:,4:6)/pi*180);
            title('Rotation Parameters (degree)');
            xlabel('number of scans'); ylabel('rotation (degree)');
            
            if exist('suptitle', 'builtin')
                suptitle(stringTitle);
            end
            
            % suptitle doesn't like the legend before, so put it after...
            axes(hs(2));
            legend('x (pitch)','y (roll)', 'z (yaw)');
            
            linkaxes(hs, 'x');
          
        else
            warning('No realignment parameters found');
        end
    case 'physio'
end