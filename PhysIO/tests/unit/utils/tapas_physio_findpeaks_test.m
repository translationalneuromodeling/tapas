function tests = tapas_physio_findpeaks_test()
% Tests whether current findpeaks function of Matlab's signal processing
% toolbox delivers same results as previous version used in reference data
%
%    tests = tapas_physio_findpeaks_test()
%
% This relies on the GE/PPU3T dataset which is particularly challenging in
% terms of peak detection because of low SNR and drift of the PPU signal
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_findpeaks_test
%
%   See also

% Author:   Lars Kasper
% Created:  2019-03-14
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

tests = functiontests(localfunctions);
end

function test_ge_ppu3t_peaks(testCase)
%% Compares previously saved cpulse (detected cardiac pulses) from 
% physio-structure to same output when re-running current version of
% GE PPU3T example
% both SPM or matlab-script based execution is possible (check parameter
% doUseSpm below!)

doUseSpm = true;

% run GE example and extract physio
pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
% TODO: Make generic!
pathExamples =  fullfile(pathPhysioPublic, '..', 'examples');

if doUseSpm
    pathCurrentExample = fullfile(pathExamples, 'GE/PPU3T');
    pathNow = pwd;
    cd(pathCurrentExample); % for prepending absolute paths correctly
    fileExample = fullfile(pathCurrentExample, 'ge_ppu3t_spm_job.mat');
    load(fileExample, 'matlabbatch');
    
    % remove unnecessary verbosity and processing of resp data
    matlabbatch{1}.spm.tools.physio.verbose.level = 0;
    matlabbatch{1}.spm.tools.physio.log_files.respiration = {''};

    spm_jobman('run', matlabbatch);
    cd(pathNow);
    
    % retrieve physio struct from saved file
    matlabbatch{1}.spm.tools.physio.model.output_physio = fullfile(pathCurrentExample, ...
        matlabbatch{1}.spm.tools.physio.save_dir{1}, ...
        matlabbatch{1}.spm.tools.physio.model.output_physio);
    load(matlabbatch{1}.spm.tools.physio.model.output_physio, 'physio');
    actPhysio = physio;
else % has verbosity...cannot switch it off
    fileExample = fullfile(pathExamples, 'GE/PPU3T/ge_ppu3t_matlab_script.m');
    run(fileExample); % will output a PhysIO=struct
    actPhysio = physio;
end


% load physio from reference data
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'utils', ...
    'physio_findpeaks_ge_ppu3t.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actSolution = actPhysio.ons_secs.cpulse;
expSolution = expPhysio.ons_secs.cpulse;

verifyEqual(testCase, actSolution, expSolution);

end

function test_ge_ppu3t_findpeaks_compatible(testCase)
%% Removes signal processing findpeaks from path and checks whether
% deprecated modified version from older Matlab release delivers same
% results
% Is also true, when no tapas_physio_findpeaks_compatible function exists

pathSignalToolbox = fileparts(which('findpeaks'));

hasCompatibleFindPeaks = ~isempty(which('tapas_physio_findpeaks_compatible'));

if ~isempty(pathSignalToolbox) && hasCompatibleFindPeaks
    rmpath(pathSignalToolbox);
    fprintf('\n\t!!! Temporarily removed %s from path to check performance vs legacy findpeaks\n', ...
        pathSignalToolbox);
    test_ge_ppu3t_peaks(testCase);
    addpath(pathSignalToolbox);
else
    assumeEqual(testCase, 1, 1);
end
end