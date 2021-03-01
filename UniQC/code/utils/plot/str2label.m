function strlabel = str2label(str)
%converts (cell of) strings into nice label for title, xlabel, Ticks etc.
%
%   strlabel = str2label(str)
%
% IN
%   str         string or cell of string
%
% OUT
%  str2label    
%
% EXAMPLE
%   str2label
%
%   See also str2fn

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