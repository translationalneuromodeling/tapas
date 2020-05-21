function filter = tapas_physio_gui_filter()
% Creates sub-part for preproc filter options in Batch Editor GUI
%
%  tapas_physio_gui_filter
%
%
%   See also
 
% Author:   Lars Kasper
% Created:  2019-07-05
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
%--------------------------------------------------------------------------
% filter_type
%--------------------------------------------------------------------------
filter_type        = cfg_menu;
filter_type.tag    = 'type';
filter_type.name   = 'Filter Type';
filter_type.help   = {'Which infinite impulse response filter shall be used?'
    '   ''Chebychev Type II (cheby2)''    Chebychev Type II filter, use for steep transition from'
    '               start to stop band'
    '   ''Butterworth (butter)''    Butterworth filter, standard filter with maximally flat'
    '               passband (Infinite impulse response), but stronger'
    '               ripples in transition band'
};
filter_type.labels = {'Chebychev Type II', 'Butterworth'};
filter_type.values = {'cheby2', 'butter'};
filter_type.val    = {'cheby2'};

%--------------------------------------------------------------------------
% filter_stopband
%--------------------------------------------------------------------------
filter_stopband         = cfg_entry;
filter_stopband.tag     = 'stopband';
filter_stopband.name    = 'Stopband';
filter_stopband.help    = {
  '[f_min, f_max] frequency interval in Hz of all frequencies, s.th. frequencies'
  '                 outside this band should definitely *NOT* pass the filter'
  '                 Default: [] '
  '                 NOTE: only relevant for ''cheby2'' filter type'
  '                 if empty, and passband is empty, no filtering is performed'
  '                 if empty, but passband exists, stopband interval is'
  '                 10% increased passband interval'
    };
filter_stopband.strtype = 'e';
filter_stopband.num     = [0 Inf];
filter_stopband.val     = {[]};

%--------------------------------------------------------------------------
% filter_passband
%--------------------------------------------------------------------------
filter_passband         = cfg_entry;
filter_passband.tag     = 'passband';
filter_passband.name    = 'Passband';
filter_passband.help    = {
  '[f_min, f_max] frequency interval in Hz of all frequency that should'
  '                 pass the passband filter'
  '                   default: [0.3 9] (to remove slow drifts, breathing'
  '                                    and slice sampling artifacts)'
  '                 if empty, no filtering is performed'
      };
filter_passband.strtype = 'e';
filter_passband.num     = [0 Inf];
filter_passband.val     = {[0.3 9]}; 

%--------------------------------------------------------------------------
%% filter_no
%--------------------------------------------------------------------------

filter_no         = cfg_branch;
filter_no.tag  = 'no';
filter_no.name = 'No';
filter_no.val  = {};
filter_no.help = {'Cardiac data remains unfiltered before pulse detection.'};

%--------------------------------------------------------------------------
%% filter_yes
%--------------------------------------------------------------------------

filter_yes      = cfg_branch;
filter_yes.tag  = 'yes';
filter_yes.name = 'Yes';
filter_yes.val  = {filter_type, filter_passband, ...
filter_stopband};
filter_yes.help = {''};


%--------------------------------------------------------------------------
%% filter
%--------------------------------------------------------------------------

filter      = cfg_choice;
filter.tag  = 'filter';
filter.name = 'Filter Raw Cardiac Time Series';
filter.val  = {filter_no};
filter.values  = {filter_no, filter_yes};
filter.help = {
    'Filter properties for bandpass-filtering of cardiac signal before peak'
    'detection, phase extraction, and other physiological traces'
 };

