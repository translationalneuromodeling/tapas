function functs = tapas_get_functions_in_sub_dirs(parDir,getFullPath)
	% Function returning the names of matlab functions in dir and subdirs.
	%
	% IN
	%	parDir 			default: pwd
	%		Starting here.
	%	getFullPath		default: true
	%		Return full path to file (otherwise just function name).
	% 
	% OUT
	%	functs
	%		Function names/paths in cell array
	%
    % Author(s): Matthias Mueller-Schrader
    % (c) Institute for Biomedical Engineering, ETH and University of Zurich, Switzerland

	% TODO: Include also .p, .mlx and .mex files


	if nargin < 1 || isempty(parDir)
		parDir = pwd;
	end

	if nargin < 2 || isempty(getFullPath)
		getFullPath = true;
	end

	ret = dir(parDir);
	names = {ret.name};
	isdirs = [ret.isdir];
	dirnames = names(isdirs);
	dirnames(strcmp(dirnames,'.')|strcmp(dirnames,'..')) = []; % get rid of '.' and '..' in unix.
	mfilenames = names(endsWith(names,'.m'));
	if ~getFullPath
		mfilenames = replace(mfilenames,'.m','');
	else
		% = {ret.folder};
		folders = cell(1,numel(mfilenames));
		folders(:) = {[ret(1).folder,filesep]};
		mfilenames = join([folders;mfilenames],'',1);
	end
	functs = mfilenames;
	for iDir = 1:numel(dirnames)
		functs = [functs,getFunctionsInSubDirs([parDir,filesep,dirnames{iDir}],getFullPath)];%#ok
	end


