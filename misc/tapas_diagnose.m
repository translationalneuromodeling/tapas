function [fList,isFine] = tapas_diagnose(toolbox_names,options)
% TAPAS_DIAGNOSE Checks if toolbox uses functions which are overloaded
%
% IN
%   toolbox_names [character vector or cell array of character vectors]
%		Toolboxed to be diagnosed. If no toolbox is specified, all toolboxes
%		are checked. This can take a while!
% 	options [struct,optional]
%		If this struct has the field(s) filterNames/notifyMultipleOccurences,
%		they are passed to tapas_diagnose_file. Compare the documentation of 
%		that function.
%
% OUT
%   fList   Filtered list of files needed to execute the function
%	isFine 	No problem found.
%
% This function is a wrapper to tapas_diagnose_file to find shadowed functions. 
% 
%
% See also tapas_diagnose_file
%
% (C) 01.03.22, Translational Neuromodeling Unit, UZH & ETH Zürich, Matthias Müller-Schrader

if nargin < 1
	toolbox_names = {}; % will default to all
end
if ~iscell(toolbox_names)
	toolbox_names = {toolbox_names};
end
if nargin < 2
	options = struct();
end 
if ~isfield(options,'filterNames')
	options.filterNames = []; % Default of tapas_diagnose_file
end 
if ~isfield(options,'notifyMultipleOccurences')
	options.notifyMultipleOccurences = [];
end

infos = tapas_get_toolbox_infos();

toolboxes_in_tapas = fieldnames(infos);
nToolbox = numel(toolboxes_in_tapas);
if isempty(toolbox_names)
	warning('This is a coffee function - get some coffee and wait for the execution to finish.')
	checkToolbox = ones(nToolbox,1,'logical');
else
	checkToolbox = ismember(toolboxes_in_tapas,toolbox_names);
	didNotFind = ~ismember(toolbox_names,toolboxes_in_tapas);
	if any(didNotFind)
        warning('I do not know the toolbox %s!\n',toolbox_names{didNotFind})
	end
end
fList = cell(0,1);
isFine = [];
for iToolbox = 1:nToolbox
	if ~checkToolbox(iToolbox)
		continue;
	end
	toolbox_name = toolboxes_in_tapas{iToolbox};
	diagnose_files = infos.(toolbox_name).diagnose_files;
	if isempty(diagnose_files)
		if ~isempty(toolbox_names)
			% This toolbox was specifically requested!
			warning('Do not have configurations to diagnose %s!',toolbox_name)
		end
		continue
	end
	fprintf('==================================================\n')
	fprintf('======= Diagnosing toolbox %s =======\n',toolbox_name)
	[fList{end+1},isFine(end+1)] = tapas_diagnose_file(diagnose_files,...
						options.filterNames,options.notifyMultipleOccurences);
end
if all(isFine)
	fprintf('Did not find a problem for the toolbox(es)\n')
	fprintf('\t%s\n',toolbox_names{:})
end

