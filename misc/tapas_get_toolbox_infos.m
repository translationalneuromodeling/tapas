function [infos, auxillary_toolboxes] = tapas_get_toolbox_infos()
	% Function storing information regarding toolboxes.
	%
	% Each toolbox is represented by a field with its lower-case-name.
	% The field has the following subfields:
	%   init_function   [character vector]
	%           If specified, use that function to initialze the toolbox (and do
	%           not anything else apart from adding init_dir to the path).
	%           If not specified, add init_dir and all subfolders 
	%   init_dir        [character vector]
	%           If init function is specified; the directory, where it can be 
	%           found (has to be added to MATLABPATH to call the function)
	%           If init function is not specified, the root dir of the toolbox,
	%           which (and all subdirs) will be added.
	%   dependencies   [character vector]
	%           Names of dependent toolboxes. Needs to be lower case (the field-
	%           name in the struct). At the moment, we cannot check for depen- 
	%           dencies of dependencies, so they should be specified as depen-
	%           dencies as well.
	%   diagnose_files [character vector/cell array of character vectors]
        %           Name(s) of files to be checked by TAPAS_diagnose('toolboxName').
	%   
	% muellmat@ethz.ch 
	% copyright (C) 2022

	infos = struct();
    %% Order determines init-order: do that by alphabet for main toolboxes
	infos = tapas_default_toolbox_info(infos, 'ceode');
    
        infos.hgf.init_function = '';
	infos.hgf.init_dir = 'HGF';
	infos.hgf.dependencies = [];
	infos.hgf.diagnose_files = '';
        infos.hgf.test_function_name = '';

	infos.huge.init_function = 'tapas_huge_compile';
	infos.huge.init_dir = 'huge';
	infos.huge.dependencies = [];
	infos.huge.diagnose_files = '';
        infos.huge.test_function_name = '';

	infos.physio.init_dir = strcat('PhysIO',filesep,'code');
	infos.physio.init_function = 'tapas_physio_init';
	infos.physio.dependencies = {'Signal Processing Toolbox', ...
        'Image Processing Toolbox', ...
        'Statistics and Machine Learning Toolbox'};
	infos.physio.diagnose_files = 'tapas_physio_main_create_regressors';
        infos.physio.test_function_name = 'tapas_physio_test';

	infos = tapas_default_toolbox_info(infos,'rDCM');
    
	infos = tapas_default_toolbox_info(infos,'sem');

	infos.uniqc.init_dir = 'UniQC';
	infos.uniqc.init_function = '';
	infos.uniqc.dependencies = [];
	infos.uniqc.diagnose_files = 'tapas_uniqc_demo_fmri_qa'; % in subfolder demo/MrSeries
        infos.uniqc.test_function_name = '';

	%% Auxillary toolboxes: 
        infos.external.init_function = '';
	infos.external.init_dir = 'external';
	infos.external.dependencies = '';
	infos.external.diagnose_files = '';
        infos.external.test_function_name = 'tapas_test_template';

    infos = tapas_default_toolbox_info(infos,'tools'); % Want to have that?
    
    auxillary_toolboxes = {'external','tools'}; % Just inform - let init decide what to do
end


function infos = tapas_default_toolbox_info(infos,folderName)
    % Setting default toolbox info (no init function, no dependencies).
    fld_name = lower(folderName);
    if ismember(fld_name,fieldnames(infos))
        warning('Infos for toolbox %s already there!\n',fld_name);
    end
    infos.(fld_name).init_function = '';
    infos.(fld_name).init_dir = folderName;
    infos.(fld_name).dependencies = '';
    infos.(fld_name).diagnose_files = '';
    infos.(fld_name).test_function_name = '';
end

