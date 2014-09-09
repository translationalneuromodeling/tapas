function tapas_physio_print_figs_to_file(verbose, save_dir)
% prints all figure handles in verbose-struct to specified filename there
%
%   physio_print_figs_to_ps(verbose)
%
% IN
%   verbose.fig_handles
%   verbose.fig_output_file
%
% OUT
%
% EXAMPLE
%   physio_print_figs_to_ps
%
%   See also
%
% Author: Lars Kasper
%           based on code by Jakob Heinzle, TNU
%
% Created: 2013-04-23
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TNU CheckPhysRETROICOR toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_print_figs_to_file.m 489 2014-05-03 12:22:54Z kasperla $
if nargin > 1
    verbose.fig_output_file = fullfile(save_dir, verbose.fig_output_file);
end

if isempty(verbose.fig_handles) && verbose.level > 0 && ...
    ~isempty(verbose.fig_output_file)
    warning('No figures found to save to file')
else
    [pfx fn sfx] = fileparts(verbose.fig_output_file);
    switch sfx
        case '.ps'
            try %level 2 PS
                print(verbose.fig_handles(1),'-dpsc2',verbose.fig_output_file);
                for k=2:length(verbose.fig_handles)
                    print(verbose.fig_handles(k),'-append',verbose.fig_output_file);
                end
            catch
                delete(verbose.fig_output_file);
                print(verbose.fig_handles(1),'-dpsc',verbose.fig_output_file);
                for k=2:length(verbose.fig_handles)
                    print(verbose.fig_handles(k),'-append',verbose.fig_output_file);
                end
            end
        case '.fig'
            for k=1:length(verbose.fig_handles)
                saveas(verbose.fig_handles(k), fullfile(pfx,[fn sprintf('_%02d', k) sfx]));
            end
        case '' % empty, do nothing!
        otherwise %'jpg', 'tiff', 'fig', ... basically everything Matlab supports via print
            switch sfx
                case {'.jpeg', '.jpg'}
                    printFormat = '-djpeg';
                case '.png'
                    printFormat = '-dpng';
                case {'.tif', '.tiff'}
                    printFormat = '-dtiffn';
                otherwise
                    printFormat = '-djpeg';
                    warning('Image format to save output figures not supported, choosing jpeg instead');
            end
            for k=1:length(verbose.fig_handles)
                print(verbose.fig_handles(k),printFormat,fullfile(pfx,[fn sprintf('_%02d', k) sfx]));
            end
    end
end
