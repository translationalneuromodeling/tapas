function [version,data] = tapas_check_online_version()
    %% Read online the current version of tapas there.
    %
    % Input
    %
    % Output
    %       version   -- Current version of tapas on github (as string)
    %       data      -- The data form the github api

    % muellmat@ethz.ch
    % copyright (C) 2020
    %
    
    % acessing the github api. matlab is converting it to a struct (data)
    data = webread('https://api.github.com/repos/translationalneuromodeling/tapas/releases');
    % depending on verbose options, we use data(i).tag_name and data(i).body
    %   for all i (if api would change).

    version = data(1).tag_name;
    version = strrep(version,'v','');

end