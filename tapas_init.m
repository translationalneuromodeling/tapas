function tapas_init()
%% Initilizes the toolbox and prints a message in the console.
%

% aponteeduardo@gmail.com
% copyright (C) 2017
%


addpath(genpath(pwd));


[version, hash] = tapas_version();
disp(strcat('Initializing TAPAS ...'));
fprintf(1, 'Version %d.%d.%d.%d\n', version{:});

tapas_print_logo();

end
