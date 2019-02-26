function tapas_rdcm_compile()
% tapas_rdcm_compile()
%
% Compiles integrator for DCM. Assumes that source code file
% 'dcm_euler_integration.c' is in the same folder as this function
% and places the compiled mex-file in the same folder.
% 
%   Input:
%
%   Output:
%
%

% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2018 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% check if mex file already exists
if ( exist('dcm_euler_integration', 'file') == 3 )
    return
end

% get path to source code
mex_path = mfilename('fullpath');
mex_path = fileparts(mex_path);

% build full path to source file
scource_path = fullfile(mex_path, 'dcm_euler_integration.c');

% check if source file present
if ( exist(scource_path,'file') ~= 2 )
    error('TAPAS:rDCM:missingSource',...
        ['Could not locate source file. Make sure dcm_euler_integration.c ' ...
         'is in the misc folder of the rDCM toolbox.']);
end

% compile the source code
mex(scource_path, '-outdir', mex_path);

end
