function test_tapas_mpdcm(fp)
%% Test the whole package 
%
% fp -- Pointer to a file for the test output, defaults to 1
%

% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
% Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
%
% Licensed under GNU General Public License 3.0 or later.
% Some rights reserved. See COPYING, AUTHORS.
%
% Revision log:
%
%

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);


tdir = mfilename('fullpath');
[tdir, fname, ~] = fileparts(tdir);

tests = dir(tdir);

for i = 1:numel(tests)
    % Avoid recursion

    [~, atest, aext] = fileparts(tests(i).name);

    if strcmp(atest, fname)
        continue;
    end

    % Avoid anything that is not a test.
    if numel(atest) < 4 || ~strcmp(atest(1:4), 'test')
        continue
    end

    if strcmp(aext, '.m')
        try
            fh = str2func(sprintf('@(x)%s(x)', atest));
            fh(fp);
        catch err
            display(getReport(err, 'extended'));
        end
    end

end

end

