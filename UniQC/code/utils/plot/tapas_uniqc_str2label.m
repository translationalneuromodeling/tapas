function strlabel = tapas_uniqc_str2label(str)
%converts (cell of) strings into nice label for title, xlabel, Ticks etc.
%
%   strlabel = tapas_uniqc_str2label(str)
%
% IN
%   str         string or cell of string
%
% OUT
%  tapas_uniqc_str2label    
%
% EXAMPLE
%   tapas_uniqc_str2label
%
%   See also tapas_uniqc_str2fn

% Author: Lars Kasper
% Created: 2013-11-07
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.

if iscell(str)
    strlabel = cell(size(str));
    for i = 1:length(str)
        strlabel{i} = regexprep(str{i}, '_', ' ');
    end
else
    strlabel = regexprep(str, '_', ' ');
end