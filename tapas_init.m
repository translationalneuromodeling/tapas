function tapas_init(varargin)    
% Initialize TAPAS and print a message in the console
%
% If no argument is given, all toolboxes are initialized.
% If you want to initialize only certain toolboxes, use their 
% names as arguments, i.e. tapas_init('hgf') for the HGF-toolbox.
% Dependent TAPAS toolboxes will be initialised as well. 
%
% To suppress startup messages, use '-noMessages'. 
% To suppress the online-checks for new revisions, use '-noUpdates'
% If startups messages are suppressed, there will be no check for updates.
% 

% muellmat@ethz.ch
% copyright (C) 2020


f = mfilename('fullpath');
[tdir, ~, ~] = fileparts(f);
addpath(tdir) % Add tapas dir to path. 
addpath(fullfile(tdir,'misc')) % Add misc for core tapas functionality.

% Separate options (start with '-' as in '-noUpdate') from toolbox names:
[init_options,toolboxes] = tapas_init_process_varargin(varargin); % 

% If not suppressed, print TAPAS logo and version:
if init_options.doShowStartupMessage
    [version, hash] = tapas_version();
    disp(strcat('Initializing TAPAS ...'));
    fprintf(1, 'Version %s.%s.%s\n', version{:});

    tapas_print_logo();
end
% Check for updates if not suppressed:
if init_options.doCheckUpdates && init_options.doShowStartupMessage    
    % The level 3 shows now the infos for all newer versions. If that is
    % too much, one might change that to 2 (only notes of newest release).
    tapas_check_for_new_release(3);
end

% This function is adding the toolboxes. If toolboxes is empty, all are added.
tapas_init_toolboxes(toolboxes,tdir,init_options.doShowStartupMessage)

% Look, if the example data folder exists. If not, give message.
if init_options.doShowStartupMessage
    if ~exist(fullfile(tdir, 'examples'), 'dir')
        fprintf(1, ...
        ['Example data can be downloaded with ' ...
        '\''tapas_download_example_data()\''\n']);
    end
end


end


function [init_options,toolboxes] = tapas_init_process_varargin(in)
% Function to separate the varargins into options (start with '-') and 
% function names. Options also have defaults.

% Separation of varagin in messages and options:
getOpts = @(x) startsWith(x,'-');
isOpt = cellfun(getOpts,in);
opts = in(isOpt);
toolboxes = in(~isOpt);
toolboxes = lower(toolboxes);

% Create struct for init options and set defaults:
init_options = struct('doShowStartupMessage',true,'doCheckUpdates',true);

% Integrate options from varargin. If unknown options are used, a warning is issued.
if ismember('-noUpdates',opts)
    init_options.doCheckUpdates = false;
    opts(ismember(opts,'-noUpdates')) = []; % delete to find wrong options
end
if ismember('-noMessages',opts)
    init_options.doShowStartupMessage = false;
    opts(ismember(opts,'-noMessages')) = []; % delete to find wrong options
end
if ~isempty(opts) 
    str = sprintf('\n\t%s',opts{:});
    warning('Received unused options %s\n I am irgnoring them!\n',str)
end

end