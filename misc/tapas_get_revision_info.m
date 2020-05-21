function [branch, revhash] = tapas_get_revision_info(directory)
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2016
%

if nargin < 1
    directory = pwd();
end

if ~(exist(directory) == 7)
    error('tapas:get_revision_info', 'no directory');
end

gitroot = fullfile(directory, '.git');

if ~(exist(gitroot) == 7)
     error('tapas:get_revision_info', 'no repository');
end

githead = fullfile(gitroot, 'HEAD');

if ~(exist(githead) == 2)
     error('tapas:get_revision_info', 'no head');
end

fp = fopen(githead, 'r');

if fp == -1
    error('tapas:get_revision_info', 'Could not read the head');
end

head = fgetl(fp);
fclose(fp);
head = regexprep(head, 'ref: ', '');
reffile = fullfile(gitroot, head);

[~, branch, ~] = fileparts(head);

if ~(exist(reffile) == 2)
    error('tapas:get_revision_info', 'Could not read the refs');
end

fp = fopen(reffile, 'r');

if fp == -1
    error('tapas:get_revision_info', 'Could not read the refs');
end

revhash = fgetl(fp);
fclose(fp);


end
