function [conn] = tapas_open_connection(config)
%% Return a connection object 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%


conn = database(config.database.database, ...
    config.database.user, ...
    config.database.password,  ...
    'org.postgresql.Driver', ...
    sprintf('jdbc:postgresql://%s:%d/', config.database.host, ...
        config.database.port));

end

