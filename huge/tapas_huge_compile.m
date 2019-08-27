function [ ] = tapas_huge_compile( )
% Compile integrator for DCM fMRI forward model. This function assumes that
% source code file 'tapas_huge_int_euler.c' is in the same folder as this
% function and places the compiled mex-file in the same folder. Running
% this function requires a C compiler to be installed. For more information
% see:
% 
% https://www.mathworks.com/support/requirements/supported-compilers.html
%

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 

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





