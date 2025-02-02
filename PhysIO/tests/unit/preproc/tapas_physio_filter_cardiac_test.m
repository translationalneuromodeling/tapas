function tests = tapas_physio_filter_cardiac_test()
% Tests whether bandpass filter on PPU example data works as expected
%
%   tests = tapas_physio_filter_cardiac_test()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_filter_cardiac_test
%
%   See also

% Author:   Lars Kasper
% Created:  2019-07-02
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auxiliary functions for automation of code folder structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path to examples, needed for all test cases
function setupOnce(testCase)
% Get PhysIO public repo base folder from this file's location
testCase.TestData.pathPhysioPublic = tapas_physio_simplify_path(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..'));
testCase.TestData.pathExamples = tapas_physio_get_path_examples(testCase.TestData.pathPhysioPublic);
% for time courses (e.g., breathing) that reach close to 0, relative
% tolerance can be misleading, use relative value to max instead
testCase.TestData.absTol = 1e-6;
end


function test_philips_ppu7t_filter_cheby2(testCase)
%% Compares previously saved Chebychev Type 2 IIR-filtered cropped cardiac 
% time course with current re-run of same batch from Philips 7T PPU data

pathCurrentExample = fullfile(testCase.TestData.pathExamples, 'Philips/PPU7T');
cd(pathCurrentExample); % for prepending absolute paths correctly
fileExample = fullfile(pathCurrentExample, 'philips_ppu7t_spm_job.m');
run(fileExample); % retrieve matlabbatch

% remove unnecessary verbosity and processing of resp data
matlabbatch{1}.spm.tools.physio.verbose.level = 0;
matlabbatch{1}.spm.tools.physio.log_files.respiration = {''};

physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);

%% Run and test for cheby2 filter
physio.preproc.cardiac.filter.type = 'cheby2';
physio.preproc.cardiac.filter.include = 1;
physio.preproc.cardiac.filter.passband = [0.5 3];
physio.preproc.cardiac.filter.stopband = [0.4 3.9];
actPhysio = tapas_physio_main_create_regressors(physio);

% load physio from reference data
fileReferenceData = fullfile(testCase.TestData.pathExamples, 'TestReferenceResults', 'preproc', ...
    'physio_filter_cardiac_cheby2.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actSolution = actPhysio.ons_secs.c;
expSolution = expPhysio.ons_secs.c;

% RelTol would be too conservative, because values close to 0 in raw
% timeseries
verifyEqual(testCase, actSolution, expSolution, ...
    'AbsTol', testCase.TestData.absTol*max(abs(expSolution)));
end

function test_philips_ppu7t_filter_butter(testCase)
%% Compares previously saved butterworth-filtered cropped cardiac time course
% with current re-run of same batch from Philips 7T PPU data

pathCurrentExample = fullfile(testCase.TestData.pathExamples, 'Philips/PPU7T');
cd(pathCurrentExample); % for prepending absolute paths correctly
fileExample = fullfile(pathCurrentExample, 'philips_ppu7t_spm_job.m');
run(fileExample); % retrieve matlabbatch

% remove unnecessary verbosity and processing of resp data
matlabbatch{1}.spm.tools.physio.verbose.level = 0;
matlabbatch{1}.spm.tools.physio.log_files.respiration = {''};

physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);


%% run and test for butterworth filter
physio.preproc.cardiac.filter.include = 1;
physio.preproc.cardiac.filter.type = 'butter';
physio.preproc.cardiac.filter.passband = [0.6 3];
physio.preproc.cardiac.filter.stopband = [];
actPhysio = tapas_physio_main_create_regressors(physio);

% load physio from reference data
fileReferenceData = fullfile(testCase.TestData.pathExamples, 'TestReferenceResults', 'preproc', ...
    'physio_filter_cardiac_butter.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actSolution = actPhysio.ons_secs.c;
expSolution = expPhysio.ons_secs.c;

% RelTol would be too conservative, because values close to 0 in raw
% timeseries
verifyEqual(testCase, actSolution, expSolution, ...
    'AbsTol', testCase.TestData.absTol*max(abs(expSolution)));
end
