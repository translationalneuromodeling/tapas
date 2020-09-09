function [example_dir] = tapas_download_example_data(version)
%% Download example data and install it in tapas/examples/.
%
% Input
%       version     -- String with the desired release. If empty it defaults
%                      to the current version.
% Output
%       example_dir - folder containing example data
%       
% Examples:
%
% Download the latest version:
% tapas_download_example_data(); 
%
% Download data from release 3.0.0
% tapas_download_example_data('3.0.0')          
%

% aponteeduardo@gmail.com
% copyright (C) 2018
%

import JavaMD5.JavaMD5
% Default to the current version
if nargin < 1
    version = tapas_get_current_version();
end

% Get the current directory
f = mfilename('fullpath');
[tdir, ~, ~] = fileparts(f);
[tapas_head, ~, ~] = fileparts(tdir);

logname = fullfile(tdir, 'log_tapas.txt');

fid = fopen(logname);
line = {};
try
    while feof(fid) == 0
        line = fgets(fid);
        line = strsplit(line);
        if strcmp(line{1}, version)
            break
        end
    end
catch err
    fclose(fid);
    rethrow(err);
end

assert(logical(numel(line)), 'tapas:example_data:unknown_version', ...
    'Version is unknonwn')
fclose(fid);

% create folder
example_dir = fullfile(tapas_head, 'examples');

if ~exist(example_dir, 'dir')
    mkdir(example_dir);
end

example_dir = fullfile(example_dir, line{1});

if exist(example_dir, 'dir')
    error('tapas:example_data:target_exists', 'Directory %s already exists.',example_dir)
end
mkdir(example_dir);

try
    % download
    zip_name = 'example.zip';
    websave(zip_name, line{2});
    % Check the md5 hash
    hash = JavaMD5(zip_name);

    if ~strcmp(hash, line{3})
        error('tapas:example_data:invalid_hash', ...
            'The downloaded file is invalid.');
    end
catch matx
    rmdir(example_dir);
    rethrow(matx);
end

unzip(zip_name, example_dir);
delete(zip_name);

end
