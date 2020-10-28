function haveNewerRelease = tapas_compare_versions(online_version,offline_version)
    %% Check if online_version is higher.
    %
    % Input
    %       online_version      -- Online version as string of numbers separated
    %                               by dots, i.e. '3.3.0'
    %       offline_version     -- Online version as string of numbers separated
    %                               by dots, i.e. '2.7.0.3'
    %
    % Output
    %       haveNewerRelease    -- true if version number of online_version is 
    %                               higher than that of offline_version

    % muellmat@ethz.ch
    % copyright (C) 2020
    %

    haveNewerRelease = false; % If both releases are the same, return false
    online_numbers = str2double(strsplit(online_version,'.'));
    offline_numbers = str2double(strsplit(offline_version,'.'));
    % Check whether they have the same lenght (oterwise fill with zeros)
    n_online = numel(online_numbers);    
    n_offline = numel(offline_numbers);
    if n_online > n_offline
        offline_numbers(n_online) = 0; % matlab fills rest with zeros
    elseif n_online < n_offline
        online_numbers(n_offline) = 0; % matlab fills rest with zeros
    end
    % Now that they have the same length, compare them number by number.
    %   Return at the first difference. 
    for ind = 1:max(n_online,n_offline)
        if online_numbers(ind) > offline_numbers(ind)
            haveNewerRelease = true;
            return;
        elseif online_numbers(ind) < offline_numbers(ind)
            haveNewerRelease = false;
            return;
        end
    end

end