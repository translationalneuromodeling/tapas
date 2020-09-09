function tapas_init_toolboxes(toolboxes,tdir,doShowMessages)
% Function initializing toolboxes
% 
% IN
%       toolboxes   
%               Names of toolboxes to add (including their dependencies).
%               If empty, all toolboxes are added. Use lower case letters.
%       tdir
%               TAPAS directory. 
%       doShowMessages
%               Whether to show messages (which toolboxes are added).
%

% muellmat@ethz.ch
% copyright (C) 2020

% Init all toolboxes, if none are specified:
doInitAll = false;

if nargin < 1 || isempty(toolboxes)
    doInitAll = true;
    toolboxes = {};
else
    toolboxes = lower(toolboxes);
    if ~iscell(toolboxes)
	   toolboxes = {toolboxes}; % We want to have it as a cell array afterwards.
    end
end
% If TAPAS dir not specified, it should be the parent dir:
if nargin < 2 || isempty(tdir)
	f = mfilename('fullpath');
	[tdir, ~, ~] = fileparts(f);
	seps = strfind(tdir,filesep);
	tdir = tdir(1:seps(end)-1);
end
% Show messages as default:
if nargin < 3
    doShowMessages = true;
end

% In the infos struct, we have all information about our toolboxes (which ones 
% we have, what other TAPAS toolboxes they depend on and whether they have a 
% init function):
infos = tapas_init_get_toolbox_infos();

% Strategy: Use fieldnames of struct for list of toolboxes. Have boolean array,
% whether to add them. 
toolbox_names = fieldnames(infos);
if doInitAll % That's easy: Just add all.
	doInit = ones(size(toolbox_names),'logical');
	toolboxes = toolbox_names;
else
	doInit = zeros(size(toolbox_names),'logical');
	for iTool = 1:numel(toolboxes) % These are to add (input argument)
		sTool = toolboxes{iTool};
		if ~ismember(sTool,toolbox_names)
			warning('I do not know the toolbox %s - skipping it.\n',sTool);
			continue; % Don't add it, if we don't know how to do that.
		end
		doInit(ismember(toolbox_names,sTool)) = true;
		dependencies = lower(infos.(sTool).dependencies);
		if ~isempty(dependencies) % TODO: Include dependencies of dependencies etc.
			doInit(ismember(toolbox_names,dependencies)) = true;
		end
	end
end
% Now that we know, which toolboxes (including dependencies) we want to add,
% we can do that.
for iTool = 1:numel(toolbox_names) % Now we are iterating over all toolboxes.
	if doInit(iTool) 
		sTool = toolbox_names{iTool};
        try
            if doShowMessages 
                % If the name of the toolbox was in the previously specified,
                % we can add it. Otherwise it was dependent.
                if ismember(sTool,toolboxes) 
                    fprintf(1,'===== Adding toolbox %s =====\n',sTool);
                else 
                    fprintf(1,'===== Adding dependent toolbox %s =====\n',sTool);
                end
            end
            % If no init function is specified, we add the "genpath(init_dir)".
            % Otherwise we call the init function.
            if isempty(infos.(sTool).init_function)
                addpath(genpath(fullfile(tdir,infos.(sTool).init_dir)));
            else
                % So that we can call the init function.
                addpath(fullfile(tdir,infos.(sTool).init_dir)); 
                % Call the init function.
                feval(infos.(sTool).init_function);
            end
        catch matx
            fprintf('Skipping toolbox %s. Initialization failed with error: "%s"\n',sTool,matx.message)
        end
	end
end
end


function infos = tapas_init_get_toolbox_infos()
    % Function returning with the dependencies and infos regarding the toolboxes
    %
    % Each toolbox is represented by a field with its lower-case-name.
    % The field has three subfields:
    %   init_function   [string]
    %           If specified, use that function to initialze the toolbox (and do
    %           not anything else apart from adding init_dir to the path).
    %           If not specified, add init_dir and all subfolders 
    %   init_dir        [string]
    %           If init function is specified; the directory, where it can be 
    %           found (has to be added to MATLABPATH to call the function)
    %           If init function is not specified, the root dir of the toolbox,
    %           which (and all subdirs) will be added.
    %   dependencies.   [string]
    %           Names of dependent toolboxes. Needs to be lower case (the field-
    %           name in the struct). At the moment, we cannot check for depen- 
    %           dencies of dependencies, so they should be specified as depen-
    %           dencies as well.
    %                  

    infos = struct();

    infos.physio.init_dir = strcat('PhysIO',filesep,'code');
    infos.physio.init_function = 'tapas_physio_init';
    infos.physio.dependencies = [];

    infos.hgf.init_function = '';
    infos.hgf.init_dir = 'HGF';
    infos.hgf.dependencies = [];

    infos.h2gf.init_function = '';
    infos.h2gf.init_dir = 'h2gf';
    infos.h2gf.dependencies = {'hgf','tools'};
    
    infos.huge.init_function = 'tapas_huge_compile';
    infos.huge.init_dir = 'huge';
    infos.huge.dependencies = [];

    infos.external.init_function = '';
    infos.external.init_dir = 'external';
    infos.external.dependencies = '';

    % The following is just a shortcut for the defaults.
    infos = tapas_init_default_toolbox_info(infos,'MICP');
    infos = tapas_init_default_toolbox_info(infos,'mpdcm');
    infos = tapas_init_default_toolbox_info(infos,'rDCM');
    infos = tapas_init_default_toolbox_info(infos,'sem');
    infos = tapas_init_default_toolbox_info(infos,'tools'); % Want to have that?
    infos = tapas_init_default_toolbox_info(infos,'VBLM');
    infos = tapas_init_default_toolbox_info(infos, 'ceode');
        
end

function infos = tapas_init_default_toolbox_info(infos,folderName)
    % Setting default toolbox info (no init function, no dependencies).
    fld_name = lower(folderName);
    if ismember(fld_name,fieldnames(infos))
        warning('Infos for toolbox %s already there!\n',fld_name);
    end
    infos.(fld_name).init_function = '';
    infos.(fld_name).init_dir = folderName;
    infos.(fld_name).dependencies = '';

end

