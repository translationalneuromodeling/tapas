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

function test_philips_ppu7t_filter_cheby2(testCase)
%% Compares previously saved Chebychev Type 2 IIR-filtered cropped cardiac 
% time course with current re-run of same batch from Philips 7T PPU data
% run GE example and extract physio

pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
pathExamples =  fullfile(pathPhysioPublic, '..', 'examples');

pathCurrentExample = fullfile(pathExamples, 'Philips/PPU7T');
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
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'preproc', ...
    'physio_filter_cardiac_cheby2.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actSolution = actPhysio.ons_secs.c;
expSolution = expPhysio.ons_secs.c;

verifyEqual(testCase, actSolution, expSolution);
end

function test_philips_ppu7t_filter_butter(testCase)
%% Compares previously saved butterworth-filtered cropped cardiac time course
% with current re-run of same batch from Philips 7T PPU data

% run GE example and extract physio
pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
% TODO: Make generic!
pathExamples =  fullfile(pathPhysioPublic, '..', 'examples');

pathCurrentExample = fullfile(pathExamples, 'Philips/PPU7T');
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
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'preproc', ...
    'physio_filter_cardiac_butter.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actSolution = actPhysio.ons_secs.c;
expSolution = expPhysio.ons_secs.c;

verifyEqual(testCase, actSolution, expSolution);
end
