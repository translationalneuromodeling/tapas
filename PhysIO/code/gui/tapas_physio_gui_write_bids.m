function write_bids = tapas_physio_gui_write_bids()
% Creates sub-part for write_bids options in Batch Editor GUI
%
%  tapas_physio_gui_write_bids
%
%
%   See also

% Author:   Johanna Bayer
% Created:  2024-27-03
% Copyright (C) 2024
%
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

%--------------------------------------------------------------------------
% write bids step
%--------------------------------------------------------------------------
bids_step        = cfg_menu;
bids_step.tag    = 'bids_step';
bids_step.name   = 'Processing step for BIDS conversion';
bids_step.help   = {'At which state of preprocessing should the bids file be written?'
'   none    (0) - No BIDS files written'
'   raw     (1) - Write raw vendor file data to BIDS'
'   norm    (2) - Write padded and normalized time series'
'                 to BIDS'
'   sync    (3) - Write padded/normalized data to BIDS'
'                 including extracted scan triggers'
'   preproc (4) - Write preprocessed data to BIDS (filtering,'
'                 cropping to acquisition window, cardiac pulse detection) (DEFAULT)'
};
bids_step.labels = {'none', 'raw', 'norm', 'sync', 'preproc'};
bids_step.values = {0, 1, 2, 3, 4};
bids_step.val    = {4};

%--------------------------------------------------------------------------
% output directory
%--------------------------------------------------------------------------
bids_dir         = cfg_files;
bids_dir.tag     = 'bids_dir';
bids_dir.name    = 'BIDS directory';
bids_dir.val     = {{''}};
bids_dir.help    = {['Specify a directory where the bids output files should ' ...
    'be written to']};
bids_dir.filter  = 'dir';
bids_dir.ufilter = '.*';
bids_dir.num     = [0 1];

%--------------------------------------------------------------------------
% Bids prefix
%--------------------------------------------------------------------------
bids_prefix         = cfg_entry;
bids_prefix.tag     = 'bids_prefix';
bids_prefix.name    = 'File prefix, optional';
bids_prefix.help    = {['specify a prefix of your liking for the bids file ' ...
    'created']};
bids_prefix.strtype = 's';
bids_prefix.num     = [0 Inf];
bids_prefix.val     = {'sub-control01_task-YOURTASK_run-1'};

%--------------------------------------------------------------------------
% Compose structure
%--------------------------------------------------------------------------

write_bids      = cfg_branch;
write_bids.tag  = 'write_bids';
write_bids.name = 'write_bids';
write_bids.val  = {bids_step, bids_dir, bids_prefix};
write_bids.help = {'Specidy bids conversion parameters'};
