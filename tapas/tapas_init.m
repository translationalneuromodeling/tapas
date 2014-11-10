function tapas_init()
 
% Get the version number
fileID = fopen('README.txt', 'rt'); 
firstLine = fgets(fileID);
fclose(fileID);

disp(strcat('Initializing TAPAS',{' '},firstLine,{' '},'...'));

disp( '_________   _____                _____       _____           ____                ');
disp( '   |       |     |              |_____|     |     |         |          ');
disp( '   |       |-----|              |           |-----|          ----|        ');
disp(['   | NU    |     |LGORITHMS for |sychiatry  |     |dvancing  ____|cience      ','']);
fprintf('\n');


addpath(genpath(pwd));




end
