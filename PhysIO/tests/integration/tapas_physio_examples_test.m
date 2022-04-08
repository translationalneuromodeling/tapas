function tests = tapas_physio_examples_test()
% Tests all PhysIO examples both as matlab scripts and via batch editor and
% compares relevant parts of physio-output structure and
% multiple_regressors file to reference results, saved by
% physio_update_examples when deploying
%
%    tests = tapas_physio_examples_test()
%
% NOTE: In order to run these tests, the corresponding example data
%       for PhysIO has be downloaded via tapas_download_example_data
%       (in misc/ subfolder of tapas release)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_examples_test
%
%   See also tapas_download_example_data physio_update_examples

% Author:   Lars Kasper
% Created:  2019-08-12
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
% run GE example and extract physio
testCase.TestData.pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..');
% TODO: Make generic!
testCase.TestData.pathExamples =  fullfile(testCase.TestData.pathPhysioPublic, '..', 'examples');
end


% close all created figures from examples after each test
function teardown(testCase)
close(testCase.TestData.createdFigHandles);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Matlab-only tests (w/o SPM) start here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function test_bids_cpulse3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BIDS/CPULSE3T example using matlab only
dirExample = 'BIDS/CPULSE3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_bids_ppu3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BIDS/PPU3T example using matlab only
dirExample = 'BIDS/PPU3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_bids_ppu3t_separate_files_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BIDS/PPU3T_Separate_Files example using matlab only
dirExample = 'BIDS/PPU3T_Separate_Files';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_biopac_txt_ppu3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BioPac_txt/PPU3T example using matlab only
dirExample = 'BioPac_txt/PPU3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_ge_ppu3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of GE/PPU3T example using matlab only
dirExample = 'GE/PPU3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ecg3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/ECG3T example using matlab only
dirExample = 'Philips/ECG3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ecg3t_v2_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/ECG3T_V2 example using matlab only
dirExample = 'Philips/ECG3T_V2';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ecg7t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/ECG7T example using matlab only
dirExample = 'Philips/ECG7T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ppu3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/PPU3T example using matlab only
dirExample = 'Philips/PPU3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_hcp_ppu3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_HCP/PPU3T example using matlab only
dirExample = 'Siemens_HCP/PPU3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vb_ecg3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VB/ECG3T example using matlab only
dirExample = 'Siemens_VB/ECG3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vb_ppu3t_sync_first_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VB/PPU3T example using matlab only
% synced to first DICOM volume of run
dirExample = 'Siemens_VB/PPU3T_Sync_First';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vb_ppu3t_sync_last_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VB/PPU3T example using matlab only
% synced to last DICOM volume of run
dirExample = 'Siemens_VB/PPU3T_Sync_Last';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vd_ppu3t_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VD/PPU3T example using matlab only
dirExample = 'Siemens_VD/PPU3T';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_siemens_vd_ppu3t_for_bids_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VD/PPU3T_For_BIDS example using matlab only
dirExample = 'Siemens_VD/PPU3T_For_BIDS';
doUseSpm = false;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_siemens_vd_ppu3t_for_bids_vs_bids_converted_matlab_only(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% from same Siemens VD data externally converted to BIDS
% to current output of re-run of Siemens_VD/PPU3T_For_BIDS example using matlab only
% !!!TODO: FIX Results, not equivalent to external BIDS converter results
dirExample = 'Siemens_VD/PPU3T_For_BIDS';
dirRefResults = 'BIDS/PPU3T_Separate_Files';
doUseSpm = false;
idxTests = []; % empty for now, should be 1:5, or at least 4:5
run_example_and_compare_reference(testCase, dirExample, doUseSpm, ...
    dirRefResults, idxTests)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SPM-requiring tests start here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function test_bids_cpulse3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BIDS/CPULSE3T example using SPM Batch Editor
dirExample = 'BIDS/CPULSE3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_bids_ppu3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BIDS/PPU3T example using SPM Batch Editor
dirExample = 'BIDS/PPU3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_bids_ppu3t_separate_files_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BIDS/PPU3T_Separate_Files example using SPM Batch Editor
dirExample = 'BIDS/PPU3T_Separate_Files';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_biopac_txt_ppu3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of BioPac_txt/PPU3T example using SPM Batch Editor
dirExample = 'BioPac_txt/PPU3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_ge_ppu3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of GE/PPU3T example using SPM Batch Editor
dirExample = 'GE/PPU3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ecg3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/ECG3T example using SPM Batch Editor
dirExample = 'Philips/ECG3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ecg3t_v2_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/ECG3T_V2 example using SPM Batch Editor
dirExample = 'Philips/ECG3T_V2';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ecg7t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/ECG7T example using SPM Batch Editor
dirExample = 'Philips/ECG7T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_philips_ppu3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Philips/PPU3T example using SPM Batch Editor
dirExample = 'Philips/PPU3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_hcp_ppu3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_HCP/PPU3T example using SPM Batch Editor
dirExample = 'Siemens_HCP/PPU3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vb_ecg3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VB/ECG3T example using SPM Batch Editor
dirExample = 'Siemens_VB/ECG3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_siemens_vb_ppu3t_sync_first_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VB/PPU3T example
% (sync to first DICOM volume time stamp) using SPM Batch Editor
dirExample = 'Siemens_VB/PPU3T_Sync_First';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_siemens_vb_ppu3t_sync_last_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VB/PPU3T example
% (sync to last DICOM volume time stamp) using SPM Batch Editor
dirExample = 'Siemens_VB/PPU3T_Sync_Last';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vd_ppu3t_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VD/PPU3T example using SPM Batch Editor
dirExample = 'Siemens_VD/PPU3T';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end


function test_siemens_vd_ppu3t_for_bids_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of Siemens_VD/PPU3T_For_BIDS example using SPM Batch Editor
dirExample = 'Siemens_VD/PPU3T_For_BIDS';
doUseSpm = true;
run_example_and_compare_reference(testCase, dirExample, doUseSpm)
end

function test_siemens_vd_ppu3t_for_bids_vs_bids_converted_with_spm(testCase)
%% Compares previously saved physio-structure and multiple regressors file
% from same Siemens VD data externally converted to BIDS
% to current output of re-run of Siemens_VD/PPU3T_For_BIDS example using SPM Batch Editor
% !!!TODO: FIX Results, not equivalent to external BIDS converter results
dirExample = 'Siemens_VD/PPU3T_For_BIDS';
dirRefResults = 'BIDS/PPU3T_Separate_Files';
doUseSpm = true;
idxTests = []; % empty for now, should be 1:5, or at least 4:5
run_example_and_compare_reference(testCase, dirExample, doUseSpm, ...
    dirRefResults, idxTests)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auxiliary Functions for automation and code re-use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function run_example_and_compare_reference(testCase, dirExample, doUseSpm, ...
    dirRefResults, idxTests)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of example in specified example sub-folder
% Note: both SPM or matlab-script based execution is possible
% (check parameter doUseSpm below!)
%
% IN
%   testCase    TestCase object, required by Matlab in this test function
%   dirExample  subfolder of examples where example data is stored
%               e.g., BIDS/PPU3T
%   doUseSpm    if true, _spm_job.mat versions of example scripts are used, using batch editor
%               if false, _matlab_script.m versions of example scripts are
%               used
%   dirRefResults
%               sub-folder of examples/TestReferenceResults that is used to
%               load expected reference solutions for all test cases
%               default: dirExample is used
%               if you want to cross-check two different integrations based
%               on the same data, this might be useful to cross-reference
%               expected test results (e.g., converted-to-BIDS data vs native
%               vendor read-in)
%   idxTests    Tests to be included in verification
%               sometimes it is necessary to skip some tests, if no
%               equivalency is expected (e.g., using the same data window,
%               but from a shorter logfile, ons_secs.raw will differ)
%               default: [1 2 3 4 5] (all)
%
% OUT
%

if nargin < 5
    idxTests = 1:5;
end

if nargin < 4
    dirRefResults = dirExample;
end

% hard-coded relative tolerance
relTol = 0.01; % 0.01 means 1 percent deviation from expected value allowed

%% Generic settings
% methods for recursively comparing structures, see
% https://ch.mathworks.com/help/matlab/ref/matlab.unittest.constraints.structcomparator-class.html
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.AbsoluteTolerance
import matlab.unittest.constraints.RelativeTolerance
import matlab.unittest.constraints.StructComparator
import matlab.unittest.constraints.NumericComparator
import matlab.unittest.constraints.StringComparator


pathExamples = testCase.TestData.pathExamples;
pathCurrentExample = fullfile(pathExamples, dirExample);

%% Actual run of example, via batch editor or as matlab script
if doUseSpm
    pathNow = pwd;
    cd(pathCurrentExample); % for prepending absolute paths correctly
    
    fileJobMat = [regexprep(lower(dirExample), '/', '_') '_spm_job.mat'];
    fileExample = fullfile(pathCurrentExample, fileJobMat);
    load(fileExample, 'matlabbatch');
    
    % remove unnecessary verbosity for test
    matlabbatch{1}.spm.tools.physio.verbose.level = 0;
    
    spm_jobman('run', matlabbatch);
    cd(pathNow);
    
    dirExampleOutput =  matlabbatch{1}.spm.tools.physio.save_dir{1};
    fileExampleOutputPhysio = matlabbatch{1}.spm.tools.physio.model.output_physio;
    fileExampleOutputTxt = matlabbatch{1}.spm.tools.physio.model.output_multiple_regressors;
    
else % has verbosity...cannot switch it off
    
    fileJobMScript = [regexprep(lower(dirExample), '/', '_') '_matlab_script.m'];
    fileExample = fullfile(pathExamples, dirExample, fileJobMScript);
    
    % runs example Matlab script
    % will output a PhysIO-struct, from which we can parse output files
    run(fileExample);
    
    % retrieve output files, remove preprending path
    [~, dirExampleOutput] = fileparts(physio.save_dir);
    [~, fn,ext] = fileparts(physio.model.output_physio);
    fileExampleOutputPhysio = [fn ext];
    [~, fn, ext] = fileparts(physio.model.output_multiple_regressors);
    fileExampleOutputTxt = [fn ext];
end


%% Retrieve current results from file
pathExampleOutput = fullfile(pathCurrentExample, dirExampleOutput);

load(fullfile(pathExampleOutput, fileExampleOutputPhysio), 'physio');
R = load(fullfile(pathExampleOutput,fileExampleOutputTxt));
actPhysio = physio;
actRegressorsFromTxt = R;

% for later closing
testCase.TestData.createdFigHandles = physio.verbose.fig_handles;

%% Load reference data and compare to actual run for certain subfields

% Load physio structure from reference data
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'examples', ...
    dirRefResults, fileExampleOutputPhysio);
load(fileReferenceData, 'physio');
expPhysio = physio;


% Compare all numeric sub-fields of physio with some tolerance
% ons_secs has all the computed preprocessed physiological and scan timing
% sync data, from which .model derives the physiological regressors later
% on
doTestOnsSecsRaw            = ismember(1, idxTests);
doTestSpulse                = ismember(2, idxTests);
doTestOnsSecs               = ismember(3, idxTests);
doTestMultipleRegressorsMat = ismember(4, idxTests);
doTestMultipleRegressorsTxt = ismember(5, idxTests);

% 1. Test ons_secs.raw only to check whether data read-in and basic
% filtering before cropping at least worked!
if doTestOnsSecsRaw
    testCase.verifyThat(actPhysio.ons_secs.raw, ...
        IsEqualTo(expPhysio.ons_secs.raw,  ...
        'Using', StructComparator(NumericComparator, 'Recursively', true), ...
        'Within', RelativeTolerance(relTol), ...
        'IgnoringFields',  {'spulse_per_vol'}...
        ), 'Comparing all numeric subfields of ons_secs.raw to check read-in and basic filtering of phys recordings');
end

% 2. Check some crucial timing parameters more vigorously
if doTestSpulse
    verifyEqual(testCase, actPhysio.ons_secs.raw.spulse, expPhysio.ons_secs.raw.spulse, ...
        'RelTol', relTol/10, ...
        'Comparing spulse (onset time of slice pulse (scan) events)');
end

% 3. Test other fields of ons_secs populated during preprocessing and some modeling steps
% Note: spulse_per_vol cannot be compared, because cell!
if doTestOnsSecs
    testCase.verifyThat(actPhysio.ons_secs, ...
        IsEqualTo(expPhysio.ons_secs,  ...
        'Using', StructComparator(NumericComparator, 'Recursively', true), ...
        'Within', RelativeTolerance(relTol), ...
        'IgnoringFields',  {'spulse_per_vol', 'raw'}...
        ), 'Comparing all numeric subfields of ons_secs to check full preprocessing of phys recordings');
end

% recursive with string
% testCase.verifyThat(actPhysio, ...
%     IsEqualTo(expPhysio, 'Using', ...
%     StructComparator(StringComparator, 'Recursively', true), ...
%     'IgnoringCase', true, ...
%     'IgnoringWhitespace', true ...
%     ));


% 4. Compare final multiple regressor matrix in physio.mat structure
% Check Multiple_Regressors output
if doTestMultipleRegressorsMat
    actSolution = actPhysio.model.R;
    expSolution = expPhysio.model.R;
    
    verifyEqual(testCase, actSolution, expSolution, ...
        'RelTol', relTol, ...
        'Comparing multiple regressors in physio.model.R');
end

%5. Test: Load reference data from multiple regressors txt file and test as
%well

if doTestMultipleRegressorsTxt
    fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'examples', ...
        dirRefResults, fileExampleOutputTxt);
    R = load(fileReferenceData);
    expRegressorsFromTxt = R;
    
    verifyEqual(testCase, actRegressorsFromTxt, expRegressorsFromTxt, ...
        'RelTol', relTol, ...
        'Comparing multiple regressors in txt-files');
end

end