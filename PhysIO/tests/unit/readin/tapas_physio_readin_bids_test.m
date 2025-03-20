function tests = tapas_physio_readin_bids_test()
% Checks whether BIDS reader still works the same on example cpulse and raw
% ppu3T trace signal
%
%   tests = tapas_physio_readin_bids_test()
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_readin_bids_test
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

%% Setup path to examples, needed for all test cases
function setupOnce(testCase)
testCase.TestData.pathExamples = tapas_physio_get_path_examples();
testCase.TestData.createdFigHandles = [];
% for time courses (e.g., breathing) that reach close to 0, relative
% tolerance can be misleading, use relative value to max instead
testCase.TestData.absTol = 1e-6;
end


%% Close all created figures from examples after each test
function teardown(testCase)
close(testCase.TestData.createdFigHandles);
end

%% Compare raw read-in PPU trace data from BIDS example to previously saved reference
% results
function test_readin_bids_ppu3t(testCase)

pathExamples = testCase.TestData.pathExamples;

% load SPM matlabbatch, but convert to pure script before executing
% remove unnecessary (beyond read-in) part from job exeuction (e.g.
% visualization, modeling)
% ...still needs SPM at the moment for conversion
% TODO: parse matlab-script.m file apart from last, execution line

pathCurrentExample = fullfile(pathExamples, 'BIDS/PPU3T');
pathNow = pwd;
cd(pathCurrentExample); % for prepending absolute paths correctly
fileExample = fullfile(pathCurrentExample, 'bids_ppu3t_spm_job.mat');
load(fileExample, 'matlabbatch');

physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);
physio.verbose.level = 0;
physio.preproc.cardiac.initial_cpulse_select.method = 'load_from_logfile'; % faster
physio = tapas_physio_main_create_regressors(physio);
cd(pathNow)

actPhysio = physio;

% load physio from reference data
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'readin', ...
    'physio_readin_bids_ppu3t.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actRaw = actPhysio.ons_secs.raw;
expRaw = expPhysio.ons_secs.raw;

verifyEqual(testCase, actRaw.t, expRaw.t, 'RelTol', 1e-6, 'Raw time vector does not match');
verifyEqual(testCase, actRaw.c, expRaw.c, 'RelTol', 1e-6, 'Raw cardiac trace does not match');
verifyEqual(testCase, actRaw.r, expRaw.r, 'RelTol', 1e-6, 'Raw respiratory trace does not match');

end


%% Compare raw read-in CPULSE data from BIDS example to previously saved reference
% results
function test_readin_bids_cpulse3t(testCase)

% run BIDS cpulse3t example and extract physio
pathExamples = testCase.TestData.pathExamples;

% load SPM matlabbatch, but convert to pure script before executing
% remove unnecessary (beyond read-in) part from job exeuction (e.g.
% visualization, modeling)
% ...still needs SPM at the moment for conversion
% TODO: parse matlab-script.m file apart from last, execution line

pathCurrentExample = fullfile(pathExamples, 'BIDS/CPULSE3T');
pathNow = pwd;
cd(pathCurrentExample); % for prepending absolute paths correctly
fileExample = fullfile(pathCurrentExample, 'bids_cpulse3t_spm_job.mat');
load(fileExample, 'matlabbatch');

physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);
physio.verbose.level = 0;
% Some modeling has to be done, otherwise no raw data preprocessed
%physio.model.retroicor.include = 0;
physio.model.retroicor.order.cr = 0;
physio.model.retroicor.order.r = 0;
physio.model.hrv.include = 0;
physio.model.rvt.include = 0;
physio = tapas_physio_main_create_regressors(physio);
cd(pathNow)

actPhysio = physio;

% load physio from reference data
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'readin', ...
    'physio_readin_bids_cpulse3t.mat');
load(fileReferenceData, 'physio');
expPhysio = physio;

% extract cpulse from actual and expected solution and compare
actRaw = actPhysio.ons_secs.raw;
expRaw = expPhysio.ons_secs.raw;

verifyEqual(testCase, actRaw.t, expRaw.t, 'Raw time vector does not match');
verifyEqual(testCase, actRaw.cpulse, expRaw.cpulse, 'Raw cardiac trace does not match');
verifyEqual(testCase, actRaw.r, expRaw.r, 'Raw respiratory trace does not match');

end

%% Compare newly written bids output header (.json) file from the Phillips ECG V3 test case to
% saved reference .json file
function test_write2bids_json(testCase)

pathExamples = testCase.TestData.pathExamples;

dirCurrentExample = 'Philips/ECG3T_V2';

% location where the reference files are stored - step norm
pathReferenceFiles = fullfile(pathExamples, 'TestReferenceResults', 'examples', dirCurrentExample);

% location of the physio example file that will be passed to create_main_regrssors
pathCurrentExample = fullfile(pathExamples, dirCurrentExample);
pathNow = pwd;
cd(pathCurrentExample); % for prepending absolute paths correctly
fileExample = fullfile(pathCurrentExample, 'philips_ecg3t_v2_spm_job.mat');
load(fileExample, 'matlabbatch');

physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);
physio.verbose.level = 0;

% minimize model, we only need some output
physio.model.retroicor.order.cr = 0;
physio.model.retroicor.order.r = 0;
physio.model.hrv.include = 0;
physio.model.rvt.include = 0;
physio = tapas_physio_main_create_regressors(physio);
cd(pathNow)

% read json file from example data

% Determine filename of physio output file
switch physio.write_bids.bids_step
    case 1
        tag = "raw";
    case 2
        tag = "norm";
    case 3
        tag = "sync";
    case 4
        tag = 'preproc';
end
bids_prefix = physio.write_bids.bids_prefix;
jsonFilename = sprintf('%2$s_desc-%1$s_physio.json',tag, bids_prefix);
actualJsonFile = fullfile(physio.write_bids.bids_dir, ...
    jsonFilename);

str = fileread(actualJsonFile); % dedicated for reading files as text
actualJson = jsondecode(str);

expectedJsonFile = fullfile(pathReferenceFiles, jsonFilename);

str = fileread(expectedJsonFile); % dedicated for reading files as text
expectedJson = jsondecode(str);

verifyEqual(testCase, actualJson, expectedJson, 'BIDS-writer JSON files do not match');

end

%% Compare newly written bids output (.tsv) file from the Phillips ECG V3 test case to
% saved reference .tsv file
function test_write2bids_tsv(testCase)

pathExamples = testCase.TestData.pathExamples;

dirCurrentExample = 'Philips/ECG3T_V2';

% location where the reference files are stored - step norm
pathReferenceFiles = fullfile(pathExamples, 'TestReferenceResults', 'examples', dirCurrentExample);

% location of the physio example file that will be passed to create_main_regrssors
pathCurrentExample = fullfile(pathExamples, dirCurrentExample);
pathNow = pwd;
cd(pathCurrentExample); % for prepending absolute paths correctly
fileExample = fullfile(pathCurrentExample, 'philips_ecg3t_v2_spm_job.mat');
load(fileExample, 'matlabbatch');

physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);
physio.verbose.level = 0;


% minimize model, we only need some output
physio.model.retroicor.order.cr = 0;
physio.model.retroicor.order.r = 0;
physio.model.hrv.include = 0;
physio.model.rvt.include = 0;
physio = tapas_physio_main_create_regressors(physio);
cd(pathNow)

% read json file from example data

% Determine filename of physio output file
switch physio.write_bids.bids_step
    case 1
        tag = "raw";
    case 2
        tag = "norm";
    case 3
        tag = "sync";
    case 4
        tag = 'preproc';
end
bids_prefix = physio.write_bids.bids_prefix;
tsvFilename = sprintf('%2$s_desc-%1$s_physio.tsv',tag, bids_prefix);
actualTsvFile = fullfile(physio.write_bids.bids_dir, ...
    tsvFilename);

% unzip compressed tab-separated values file and read into matlab variable
gunzip([actualTsvFile, '.gz']);
actualTsv = load(actualTsvFile, 'ascii');

expectedTsvFile = fullfile(pathReferenceFiles, tsvFilename);
gunzip([expectedTsvFile, '.gz']);
expectedTsv = load(expectedTsvFile, 'ascii');

verifyEqual(testCase, actualTsv, expectedTsv, 'BIDS-writer TSV files do not match');

% delete unzipped files after testing completed
delete(expectedTsvFile);
delete(actualTsvFile);

end
