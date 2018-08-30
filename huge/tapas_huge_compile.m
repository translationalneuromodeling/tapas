%% [ ] = tapas_huge_compile( )
%
% Compiles integrator for DCM. Assumes that source code file
% 'tapas_huge_int_euler.c' is in the same folder as this function and
% places the compiled mex-file in the same folder.
%


% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% https://github.com/translationalneuromodeling/tapas/issues
%
function [ ] = tapas_huge_compile( )
%%
% check if mex file already exists
if exist('tapas_huge_int_euler', 'file') == 3
    return
end

% get path to toolbox
tbPath = mfilename('fullpath');
tbPath = fileparts(tbPath);

% build full path to source file
scPath = fullfile(tbPath, 'tapas_huge_int_euler.c');

% check if source file present
if exist(scPath,'file')~=2
    error('TAPAS:HUGE:missingSource',...
        ['Could not locate source file. Make sure ' ...
         'tapas_huge_int_euler.c is in the huge toolbox folder.']);
end

% compile
try
    mex(scPath, '-outdir', tbPath);
catch err
    disp('tapas_huge_compile: Failed to compile mex function.');
    disp('Make sure you have selected a C language compiler for mex.');
    disp('For more information, enter mex -help on the command line.');
    rethrow(err)
end





