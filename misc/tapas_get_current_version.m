function [version] = tapas_get_current_version()
%% Get the current version of tapas.
%
% Input
%
% Output
%       version     -- A string with the the three version digits.

% aponteeduardo@gmail.com
% copyright (C) 2018
%

f = mfilename('fullpath');
[tdir, ~, ~] = fileparts(f);

logname = fullfile(tdir, 'log_tapas.txt');

fid = fopen(logname);
try
    line = fgets(fid);
    line = strsplit(line);
    fclose(fid);
catch err
    fclose(fid);
    rethrow(err);
end

version = line{1};

end
