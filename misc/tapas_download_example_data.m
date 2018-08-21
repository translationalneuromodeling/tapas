function [] = tapas_download_example_data(version)
%% Downloads example data and installs it in tapas/examples/.
%
% Input
%       version     -- The tapas version that is desired.
% Output
%       

% aponteeduardo@gmail.com
% copyright (C) 2018
%

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

% Check the md5 hash, for security reasons
zip_name = 'example.zip';
websave(zip_name, line{2});
hash = JavaMD5(zip_name);

if ~strcmp(hash, line{3})
    error('tapas:example_data:invalid_hash', ...
        'The downloaded file is invalid.');
end

example_dir = fullfile(tapas_head, 'examples');

if ~exist(example_dir, 'dir')
    mkdir(example_dir);
end

new_example = fullfile(example_dir, line{1});

if ~exist(new_example, 'dir')
    mkdir(new_example);
else
    error('tapas:example_data:target_exists', 'Target directory exists.')
end

unzip(zip_name, new_example);
delete(zip_name);

end
