function tapas_init()
%% Initilizes the toolbox and prints a message in the console.
%

% aponteeduardo@gmail.com
% copyright (C) 2017
%

f = mfilename('fullpath');
[tdir, ~, ~] = fileparts(f);

addpath(genpath(tdir));

[version, hash] = tapas_version();
disp(strcat('Initializing TAPAS ...'));
fprintf(1, 'Version %s.%s.%s\n', version{:});

tapas_print_logo();

% Check if the examples directory exist and print a message is required.

if ~exist(fullfile(tdir, 'examples'), 'dir')
    fprintf(1, ...
    ['Example data can be downloaded with ' ...
    '\''tapas_download_example_data()\''\n']);
end

end
