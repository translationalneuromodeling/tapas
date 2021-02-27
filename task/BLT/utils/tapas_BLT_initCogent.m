function tapas_BLT_initCogent(store)
% tapas_BLT_initCogent starts cogent which is needed for the response box
%
% About:
%   Coded by F.Petzschner 19. April 2017
%	Last change: F.Petzschner 19. April 2017


if  ~store.keyboard
config_serial(store.scanner.boxport, 19200, 0, 0, 8);
start_cogent;
end

end