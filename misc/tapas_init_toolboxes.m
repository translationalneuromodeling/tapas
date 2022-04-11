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
[infos,aux_tools] = tapas_get_toolbox_infos();

% Strategy: Use fieldnames of struct for list of toolboxes. Have boolean array,
% whether to add them. 
toolbox_names = fieldnames(infos);
if doInitAll % That's easy: Just add all.
	doInit = ones(size(toolbox_names),'logical');
	toolboxes = toolbox_names(~ismember(toolbox_names,aux_tools));
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
doInit(ismember(toolbox_names,aux_tools)) = true; % Always add these.


for iTool = 1:numel(toolbox_names) % Now we are iterating over all toolboxes.
	if doInit(iTool) 
		sTool = toolbox_names{iTool};
        try
            if doShowMessages 
                % If the name of the toolbox was in the previously specified,
                % we can add it. Otherwise it was dependent.
                if ismember(sTool,toolboxes) 
                    str = sprintf(['~~~~~~~~~~~~~~~~~~~~~~~~ ADDING TOOLBOX',...
                        ' <strong>%s</strong> ~~~~~~~~~~~~~~~~~~~~~~~~'],upper(sTool));
                    str(end+1:80+17) = '~'; % 17 for <strong></strong>
                    fprintf(1,'%s\n',str);
                else 
                    %fprintf(1,'===== Adding dependent toolbox %s =====\n',sTool);
                    str = sprintf(['~~~~~~~~~~~~~~~~~~~~~~~~ ADDING DEPENDENCY ',...
                        '%s ~~~~~~~~~~~~~~~~~~~~~~~~'],upper(sTool));
                    str(end+1:80) = '~'; 
                    fprintf(1,'%s\n',str);
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
