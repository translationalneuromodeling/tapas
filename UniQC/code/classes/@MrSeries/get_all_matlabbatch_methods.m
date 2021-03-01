function namesMatlabbatchMethods = get_all_matlabbatch_methods(this)
% Returns cell of string identifiers of all methods using a matlabbatch
%
%   Y = MrSeries()
%   Y.get_all_matlabbatch_methods()
%
% This is a method of class MrSeries.
%
% IN
%
% OUT
%   nameMatlabbatchMethods  cell(1,nMethods) of string identifiers of all 
%                           MrSeries-methods using an SPM matlabbatch
%
% EXAMPLE
%   get_all_matlabbatch_methods
%
%   See also MrSeries get_matlabbatch

% Author:   Lars Kasper
% Created:  2014-08-30
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

namesMatlabbatchMethods = {
   'compute_tissue_probability_maps'
   'coregister'
   'realign'
   'smooth'
   'specify_and_estimate_1st_level'
    }';