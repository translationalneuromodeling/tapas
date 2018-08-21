function tapas_init()
%% Initilizes the toolbox and prints a message in the console.
%

% aponteeduardo@gmail.com
% copyright (C) 2017
%


addpath(genpath(pwd));


[version, hash] = tapas_version();
disp(strcat('Initializing TAPAS ...'));
fprintf(1, 'Version %s.%s.%s\n', version{:});

tapas_print_logo();

end
