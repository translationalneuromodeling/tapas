function tapas_find_dependencies()
	% function finding dependencies of tapas toolboxes
	%
	% 
	% (C) 2020 Matthias Müller-Schrader TNU-IBT Universität Zürich & ETH Zürich
	%

	% parameters
	exclDir = {'.','..','.admin','.git','misc'};
	useAg = false; % otherwise use ack
	ignoreInds = [7,6]; %Ignore the first and last indices (strange output from ag)
	printCommands = true;

	cutInds = @(x) x(ignoreInds(1)+1:end - ignoreInds(2) - 1);
	f = mfilename('fullpath');
	[tdir, ~, ~] = fileparts(f);
	pdir = tdir(1:end-7); % '/.admin' -> is tapas folder (without last '/')
	curr_dir = pwd;
	cd(pdir);
	dr = dir(pdir);
	dr = dr([dr.isdir]); % only folders
	
	dr = dr(~ismember({dr.name},exclDir));

	nDir = numel(dr);
	needAlso = cell(1,nDir);
	for iDir = 1:nDir
		fprintf(1,'==========%s===============\n',dr(iDir).name)
		drPath = strcat(pdir,filesep,dr(iDir).name);
		functs = tapas_get_functions_in_sub_dirs(drPath,false);
		for iFun = 1:numel(functs)
			funct = functs{iFun};
			funct = strrep(funct,'.m','');
			if useAg
				cmd_s = 'ag';
			else
				cmd_s = 'ack';
			end
			cmd = sprintf('%s  %s -l -w --ignore-dir %s',cmd_s,funct,dr(iDir).name);
			% --range-start=''^sub \\w+''
			[stat,ret] = system(cmd);
			if stat && isempty(ret)
				continue;
			end
			%disp(ret)
			retc = split(ret,newline);
			if useAg
				retc = cellfun(cutInds,retc,'UniformOutput',false);
			end
			ise = cellfun(@isempty,retc);
			if all(ise)
				continue;
			end
			retc = retc(~ise);
			fprintf(1,'%s\n',funct)
			fprintf(1,'\t%s\n',retc{:});
		end
	end


	cd(curr_dir);
