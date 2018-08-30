function [branch, hash] = tapas_get_tapas_revision(verbose)
%% 
%
% Input
%
% Output
%

%
% aponteeduardo@gmail.com
% copyright (C) 2017
%

n = 0;

n = n + 1;
if nargin < n
    verbose = 1;
end

% Get current location
f = mfilename('fullpath');

[tdir, ~, ~] = fileparts(f);

tdir = fullfile(tdir, '..');
try
    [branch, hash] = tapas_get_revision_info(tdir);
catch err
    if strcmp('tapas:get_revision_info', err.identifier)
        if verbose 
            display(getReport(err))
        end
        branch = '';
        hash = '';
    else
        rethrow(err)
    end
end

end

