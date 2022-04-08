function tapas_physio_save_batch_mfile(matlabbatch, fileBatchM, ...
    doRemoveEmptyDefaults)
% Saves created matlabbatch variable to .m-file for later call of
% spm_jobman or run(fileBatchM)
%
%    tapas_physio_save_batch_mfile.m(matlabbatch, fileBatchM)
%
% IN
%   matlabbatch     variable from workspace
%   fileBatchM      m-file that we save to
%   doRemoveEmptyDefaults   
%                   if true, all lines in the batch that have empty
%                   (default) values will be removed
%                   default: false
%   
% OUT
%
% EXAMPLE
%   tapas_physio_save_batch_mfile.m
%
%   See also
 
% Author:   Lars Kasper
% Created:  2022-03-30
% Copyright (C) 2022 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
if nargin < 3
    doRemoveEmptyDefaults = false;
end

str = gencode(matlabbatch)';

if doRemoveEmptyDefaults
    indLineRemove = tapas_physio_find_string(str, ...
        {'= \[\];', '= {};', '= {''''};', '= '''';'});
    indLineRemove = cell2mat(indLineRemove);
    str(indLineRemove) = [];
end


nLines = numel(str);
fid = fopen(fileBatchM, 'w+');

for iLine = 1:nLines
    fprintf(fid, '%s\n', str{iLine});
end
fclose(fid);
