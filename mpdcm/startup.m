function startup()
%% Loads all the submodules of mpdcm 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

[current, ~, ~] = fileparts(mfilename('fullpath'));

mainfiles = fullfile(current, 'matlab');
testfiles = fullfile(current, 'test');
mexfiles = fullfile(current, 'src');

assert(isdir(mainfiles), '% is not a directory', mainfiles);
addpath(mainfiles);

assert(isdir(testfiles), '% is not a directory', testfiles);
addpath(testfiles);

assert(isdir(mexfiles), '% is not a directory', mexfiles);
addpath(mexfiles);

end
