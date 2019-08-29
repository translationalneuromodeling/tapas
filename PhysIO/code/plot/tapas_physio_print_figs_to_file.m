function verbose = tapas_physio_print_figs_to_file(verbose, save_dir)
% prints all figure handles in verbose-struct to specified filename there
%
%   verbose = tapas_physio_print_figs_to_file(verbose, save_dir)
%
% IN
%   verbose.fig_handles
%   verbose.fig_output_file
%
% OUT
%
% EXAMPLE
%   verbose = tapas_physio_print_figs_to_file(verbose, save_dir)
%
%   See also

% Author: Lars Kasper
%           based on code by Jakob Heinzle, TNU
%
% Created: 2013-04-23
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public Licence (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

if nargin > 1
    verbose.fig_output_file = fullfile(save_dir, verbose.fig_output_file);
end

if ~isfield(verbose, 'fig_handles') || numel(verbose.fig_handles) == 0 || ...
        isempty(verbose.fig_handles) && ~isempty(verbose.fig_output_file)
    if verbose.level > 0 
        tapas_physio_log('No figures found to save to file', verbose, 1);
    end
else
    [pfx fn sfx] = fileparts(verbose.fig_output_file);
    switch sfx
        case '.ps'
            try %level 2 PS
                for k=1:length(verbose.fig_handles)
                    print(verbose.fig_handles(k),'-dpsc2', '-append',verbose.fig_output_file);
                end
            catch
                delete(verbose.fig_output_file);
                for k=1:length(verbose.fig_handles)
                    print(verbose.fig_handles(k), '-dpsc', '-append', verbose.fig_output_file);
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
