function verbose = tapas_physio_close_figs(verbose)
% Close figures given argument 'verbose' with a list of figure handles
%
%   verbose = tapas_physio_close_figs(verbose)
%
% IN
%   verbose.fig_handles
%
% OUT
%
% EXAMPLE
%   verbose = tapas_physio_close_figs(verbose)
%
% Author: Lars Kasper, Stephan Heunis
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


if ~isfield(verbose, 'fig_handles') || numel(verbose.fig_handles) == 0 || isempty(verbose.fig_handles)
    if verbose.level > 0
        tapas_physio_log('No figures found to close', verbose, 1);
    end
else
    for k=1:length(verbose.fig_handles)
        close(verbose.fig_handles(k))
    end
end
