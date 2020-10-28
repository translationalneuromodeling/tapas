function haveNewerRelease = tapas_check_for_new_release(verbose)
    %% Check if a new release is available. 
    %
    % Input
    %       verbose         -- Show information on command prompt:
    %                               0   no information
    %                               1   only if there is a new release (or not)
    %                               2   as 1, but if there is a new release also
    %                                   the release notes for the lase release
    %                               3   as 1, but if there is a new release also
    %                                   the release notes for all newer
    %                                   releases.
    %
    % Output
    %       haveNewerRelease  -- True if there is a newer release on github

    % muellmat@ethz.ch
    % copyright (C) 2020
    %

    if nargin < 1
        verbose = 3;
    end

    [online_version,data] = tapas_check_online_version();
    if isempty(online_version) % Could not communicate with server
        haveNewerRelease = false;
        if verbose
            fprintf(1, 'Could not reach github to get current TAPAS version!\n');
        end
        return;
    end  
    offline_version = tapas_get_current_version();
    % both versions are strings in the form '3.3.0'. 
    haveNewerRelease = tapas_compare_versions(online_version,offline_version);

    if verbose % Show info
        if ~haveNewerRelease
            fprintf(1, '\nYour TAPAS version is up-to-date (version %s).\n',offline_version);
        else
            fprintf(1, ['\nThere is a new TAPAS release available (installed %s /'...
                +'newest %s)!\n'],offline_version,online_version);
            if verbose > 1 % Show release notes of the new releases
                try % in case the api changed and the structure of the struct is
                    % different
                    n_data = numel(data);
                    fprintf(1, 'Release notes:\n')
                    if verbose == 2
                        n_data = 1; % Just iterate once through loop
                    end
                    for i_data = 1:n_data 
                        str_version = data(i_data).tag_name;
                        str_version = strrep(str_version,'v','');
                        if tapas_compare_versions(str_version,offline_version)
                            fprintf(1, '===== TAPAS v%s =====\n',str_version)
                            body = data(i_data).body;
                            body = tapas_remove_problematic_escape_characters(body);
                            fprintf(1,body);
                            fprintf(1,'\n======================\n')
                        end
                    end
                catch 
                    fprintf(1, ['Cannot print release notes. Maybe the github API'...
                        ' has changed.\n']);
                end
            end
        end
    end

end
        

function str = tapas_remove_problematic_escape_characters(str)
    %% Removing problematic escape characters 
    %
    % Input
    %       str     -- String with problematic escape characters
    % 
    % Output 
    %       str     -- String without problematic escape characters

    % matlab does not like some escape sequences from github
    str = strrep(str,'\','');
end
