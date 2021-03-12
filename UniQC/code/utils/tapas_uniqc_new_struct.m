function tapas_uniqc_new_struct(varargin)
% Creates a new script (e.g. tests/examples) including header.
% tapas_uniqc_new_struct(funname) opens the editor and pastes the content
% of a user-defined template into the file funname.m.
% 
%   Example
%       tapas_uniqc_new_struct myscript
%           OR 
%       tapas_uniqc_new_struct myscript author
%
%   opens the editor and pastes the following 
% 
% 	% script myscript
% 	%MYSCRIPT  One-line description here, please.
% 	%
% 	%   Example
% 	%   MYSCRIPT
% 	%
% 	%   See also
% 
% 	% Author: Your name
% 	% Created: 2005-09-22
% 	% Copyright 2005 Your company.
% 
%   See also edit, mfiletemplate

% Author: Peter (PB) Bodin
% Created: 2005-09-22
% Modified: 2019-01-05 (Saskia Bollmann and Lars Kasper)	

    
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
			warning('tapas_uniqc_new_struct without argument is the same as edit')
			return;
		case 1
			fname=varargin{:};
			edit(fullfile(pwd,fname));
            authors = 'Saskia Bollmann & Lars Kasper'; %default authors, set this further down in function authors
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
            '% This is a predefined options/parameter structure for CLASS_OR_FUNCTION'
            '%'
            '%   outputStruct = $filename(input)'
            '%'
            '%'
            '%   See also CLASS_OR_FUNCTION'
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
            ''
            'properties'
            ''
            '   %INSERT_PROPERTY_DESCRIPTION_ABOVE_EACH_PROPERTY'
            ''
            'end'
            'methods'
            '   function this = $filename(varargin)'
            '   % Constructor of class, allows ''PropertyName'', ''PropertyValue'''
            '   % pairs and predefined shortcuts'
            '   % everything else set to default ([])'
            '       if nargin'
            '           for cnt = 1:nargin/2 % save them to object properties'
            '               this.(varargin{2*cnt-1}) = varargin{2*cnt};'
            '           end'
            '       end'
            '   end'
            'end'
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