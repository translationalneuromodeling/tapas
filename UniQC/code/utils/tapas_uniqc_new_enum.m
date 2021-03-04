function tapas_uniqc_new_enum(varargin)
%   tapas_uniqc_new_enum creates a new enumeration file including header using a template.
%   tapas_uniqc_new_enum(funname) opens the editor and pastes the content
%   of a user-defined template into the file @funname/funname.m.
% 
%   Example
%   tapas_uniqc_new_enum myfunfun

% Author: Peter (PB) Bodin
% Created: 2005-09-22
% Modified: 2014-04-15 (Saskia Klein and Lars Kasper, IBT Zurich)	

    
	% See the variables repstr, repwithstr and tmpl to figure out how
	% to design your own template.
	% Edit tmpl to your liking, if you add more tokens in tmpl, make
	% sure to add them in repstr and repwithstr as well.
	
	% I made this function just for fun to check out some java handles to
	% the editor. It would probably be better to fprintf the template
	% to a new file and then call edit, since the java objects might change
	% names between versions.

    switch nargin
		case 0
			edit
			warning('tapas_uniqc_new_enum without argument is the same as edit')
			return;
		case 1
			fname=varargin{:};
			edit(fullfile(pwd,fname));
            % default authors, set down in function authors
            authors = 'Saskia Bollmann & Lars Kasper';
        case 2
            fname = varargin{1};
            authors = varargin{2};
		otherwise
			error('tapas:uniqc:TooManyInputArguments', ...
                'too many input arguments')
    end

	try lasterror
		edhandle=com.mathworks.mlservices.MLEditorServices;
		
        % R2009a => 2009.0, R2009b = 2009.5
        vs = version('-release');
        v = str2double(vs(1:4));
        if vs(5)=='b'
            v = v + 0.5;
        end
           
        if v < 2009.0
            edhandle.builtinAppendDocumentText(strcat(fname,'.m'),parse(fname,authors));
        else
            edhandle.getEditorApplication.getActiveEditor.appendText(parse(fname, authors));
        end
	catch
		rethrow(lasterror)
	end

	function out = parse(func, authors)

		tmpl={ ...
			'classdef $filename'
			'%ONE_LINE_DESCRIPTION'
			'%'
            '%'
			'% EXAMPLE'
			'%   $filename'
			'%'
			'%   See also'
			' '
			'% Author:   $author'
			'% Created:  $date'
			'% Copyright (C) $year $institute'
            '%                    $company'
            '%'
            '% This file is part of the TAPAS UniQC Toolbox, which is released' 
            '% under the terms of the GNU General Public License (GPL), version 3. '
            '% You can redistribute it and/or modify it under the terms of the GPL'
            '% (either version 3 or, at your option, any later version).'
            '% For further details, see the file COPYING or'
            '%  <http://www.gnu.org/licenses/>.'
            ' '
            'enumeration'
            ' % COMMENT_BEFORE_ENUM_VALUE'
            'end % enumeration'
            ' '
            ' '
            'properties'
            ' % COMMENT_BEFORE_PROPERTY'
            'end % properties'
            ' '
            ' '
            ' '
            'methods'
            ''
            '% Constructor of class, only needed, if properties initialized'
            'function this = $filename()'
            'end'
            ''
            '% NOTE: Useful methods are: char, eq (==) and ne (~=).'
            ''
            'end % methods'
            ' '
            'end'
            };

		repstr={...
			'$filename'
			'$FILENAME'
			'$date'
			'$year'
			'$author'
            '$institute'
			'$company'};

		repwithstr={...
			func
			upper(func)
			datestr(now,29)
			datestr(now,10)
			authors
			'Institute for Biomedical Engineering'
            'University of Zurich and ETH Zurich'};

		for k = 1:numel(repstr)
			tmpl = strrep(tmpl,repstr{k},repwithstr{k});
		end
		out = sprintf('%s\n',tmpl{:});
	end
end