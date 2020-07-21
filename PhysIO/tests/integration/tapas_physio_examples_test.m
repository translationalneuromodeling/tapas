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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Auxiliary Functions for automation and code re-use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function run_example_and_compare_reference(testCase, dirExample, doUseSpm)
%% Compares previously saved physio-structure and multiple regressors file
% to current output of re-run of example in specified example sub-folder
% Note: both SPM or matlab-script based execution is possible
% (check parameter doUseSpm below!)

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

%% 1. Test: Load reference data from multiple regressors file
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'examples', ...
    dirExample, fileExampleOutputTxt);
R = load(fileReferenceData);
expRegressorsFromTxt = R;

verifyEqual(testCase, actRegressorsFromTxt, expRegressorsFromTxt, ...
     'RelTol', relTol, ...
     'Comparing multiple regressors in txt-files');


%% 2. load physio structure from reference data
fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'examples', ...
    dirExample, fileExampleOutputPhysio);
load(fileReferenceData, 'physio');
expPhysio = physio;

% Check Multiple_Regressors output
actSolution = actPhysio.model.R;
expSolution = expPhysio.model.R;

verifyEqual(testCase, actSolution, expSolution, ...
    'RelTol', relTol, ...
    'Comparing multiple regressors in physio.model.R');


%% 3. compare all numeric sub-fields of physio with some tolerance

% ons_secs has all the computed preprocessed physiological and scan timing
% sync data, from which .model derives the physiological regressors later
% on
% spulse_per_vol cannot be compared, because cell!
testCase.verifyThat(actPhysio.ons_secs, ...
    IsEqualTo(expPhysio.ons_secs,  ...
    'Using', StructComparator(NumericComparator, 'Recursively', true), ...
    'Within', RelativeTolerance(relTol), ...
    'IgnoringFields',  {'spulse_per_vol'}...
    ), 'Comparing all numeric subfields of ons_secs to check preprocessing of phys recordings');

% recursive with string
% testCase.verifyThat(actPhysio, ...
%     IsEqualTo(expPhysio, 'Using', ...
%     StructComparator(StringComparator, 'Recursively', true), ...
%     'IgnoringCase', true, ...
%     'IgnoringWhitespace', true ...
%     ));

end