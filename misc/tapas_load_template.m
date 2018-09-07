function [template] = tapas_load_template(fname)
%% Loads a file that work as a template for filling. 
%
% Input
%   fname       -- File name.
%       
% Output
%   string      -- An array containing the string. 
%       

% aponteeduardo@gmail.com
% copyright (C) 2017
%

fid = fopen(fname);
try
    template = {};
    while feof(fid) == 0
       template{end + 1} = fgets(fid);
    end
catch err
    % Make sure that the file is closed properly.
    try
        fclose(fid);
    end
    rethrow(err);
end
fclose(fid);
template = [template{:}];

end

