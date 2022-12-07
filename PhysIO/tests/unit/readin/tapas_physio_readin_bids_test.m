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

% % compare raw read-in PPU trace data from BIDS example to previously saved reference
% % results
% function test_readin_bids_ppu3t(testCase)
% 
% % Get PhysIO public repo base folder from this file's location
% pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
%pathExamples = tapas_physio_get_path_examples(pathPhysioPublic);
% 
% 
% % load SPM matlabbatch, but convert to pure script before executing
% % remove unnecessary (beyond read-in) part from job exeuction (e.g.
% % visualization, modeling)
% % ...still needs SPM at the moment for conversion
% % TODO: parse matlab-script.m file apart from last, execution line
% 
% pathCurrentExample = fullfile(pathExamples, 'BIDS/PPU3T');
% pathNow = pwd;
% cd(pathCurrentExample); % for prepending absolute paths correctly 
% fileExample = fullfile(pathCurrentExample, 'bids_ppu3t_spm_job.mat');
% load(fileExample, 'matlabbatch');
% 
% physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);
% physio.verbose.level = 0;
% physio.preproc.cardiac.initial_cpulse_select.method = 'load_from_logfile'; % faster
% physio = tapas_physio_main_create_regressors(physio);
% cd(pathNow)
% 
% actPhysio = physio;
% 
% % load physio from reference data
% fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'readin', ...
%     'physio_readin_bids_ppu3t.mat');
% load(fileReferenceData, 'physio');
% expPhysio = physio;
% 
% % extract cpulse from actual and expected solution and compare
% actRaw = actPhysio.ons_secs.raw;
% expRaw = expPhysio.ons_secs.raw;
% 
% verifyEqual(testCase, actRaw.t, expRaw.t, 'RelTol', 1e-6, 'Raw time vector does not match');
% verifyEqual(testCase, actRaw.c, expRaw.c, 'RelTol', 1e-6, 'Raw cardiac trace does not match');
% verifyEqual(testCase, actRaw.r, expRaw.r, 'RelTol', 1e-6, 'Raw respiratory trace does not match');
% 
% end
% 
% 
% % compare raw read-in CPULSE data from BIDS example to previously saved reference
% % results
% function test_readin_bids_cpulse3t(testCase)
% 
% % run BIDS cpulse3t example and extract physio
% pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
% pathExamples = tapas_physio_get_path_examples(pathPhysioPublic);
% 
% % load SPM matlabbatch, but convert to pure script before executing
% % remove unnecessary (beyond read-in) part from job exeuction (e.g.
% % visualization, modeling)
% % ...still needs SPM at the moment for conversion
% % TODO: parse matlab-script.m file apart from last, execution line
% 
% pathCurrentExample = fullfile(pathExamples, 'BIDS/CPULSE3T');
% pathNow = pwd;
% cd(pathCurrentExample); % for prepending absolute paths correctly 
% fileExample = fullfile(pathCurrentExample, 'bids_cpulse3t_spm_job.mat');
% load(fileExample, 'matlabbatch');
% 
% physio = tapas_physio_job2physio(matlabbatch{1}.spm.tools.physio);
% physio.verbose.level = 0;
% % Some modeling has to be done, otherwise no raw data preprocessed
% %physio.model.retroicor.include = 0;
% physio.model.retroicor.order.cr = 0;
% physio.model.retroicor.order.r = 0;
% physio.model.hrv.include = 0;
% physio.model.rvt.include = 0;
% physio = tapas_physio_main_create_regressors(physio);
% cd(pathNow)
% 
% actPhysio = physio;
% 
% % load physio from reference data
% fileReferenceData = fullfile(pathExamples, 'TestReferenceResults', 'readin', ...
%     'physio_readin_bids_cpulse3t.mat');
% load(fileReferenceData, 'physio');
% expPhysio = physio;
% 
% % extract cpulse from actual and expected solution and compare
% actRaw = actPhysio.ons_secs.raw;
% expRaw = expPhysio.ons_secs.raw;
% 
% verifyEqual(testCase, actRaw.t, expRaw.t, 'Raw time vector does not match');
% verifyEqual(testCase, actRaw.cpulse, expRaw.cpulse, 'Raw cardiac trace does not match');
% verifyEqual(testCase, actRaw.r, expRaw.r, 'Raw respiratory trace does not match');
% 
% end

% compare  newly written bids output file from the Phillips ECG V3 test case to 
% saved files

function test_compare_write2bids_consistency(testCase)
  pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
  
  pathExamples = tapas_physio_get_path_examples(pathPhysioPublic);

   % location where the reference files are stored - step norm
   pathReferenceFiles = fullfile(pathExamples, 'TestReferenceResults', 'examples','write2bids', 'norm');

   % location of the physio example file that will be passed to create_main_regrssors
   % pathExampleData = fullfile(pathExamples, 'write2bids', 'norm');
   
   pathExampleData = fullfile(pathExamples, 'Philips', 'ECG3T_V2');
   load(fullfile(pathExampleData,'physio_out', 'physio.mat'), 'physio'); % this physio structure contains data from step 2
    
    % does that work?
    cd(fullfile(pathExampleData))
    physio = tapas_physio_main_create_regressors(physio);

    % read json file from example data

    % go on here
    fileName = './physio_out/sub-01_task_desc_physio_norm.json'; % filename in JSON extension 
    str = fileread(fileName); % dedicated for reading files as text 
    ExampleJson = jsondecode(str);

    % read json file from reference folder
    cd(fullfile(pathReferenceFiles))

    fileName = 'sub-01_task_desc_physio_norm.json'; % filename in JSON extension 
    str = fileread(fileName); % dedicated for reading files as text 
    ReferenceJson = jsondecode(str);
    
    verifyEqual(testCase, ExampleJson, ReferenceJson, 'json files do not match');

end 