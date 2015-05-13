function test_mpdcm_get_device(fp)
%% Test mpdcm_get_device
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
%
% Author: Eduardo Aponte, TNU, UZH & ETHZ - 2015
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


% Test whether there is any clear bug
try
    d = mpdcm_get_device();
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
end


end

