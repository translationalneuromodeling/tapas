function test_mpdcm_fmri_mle(fp)
%% Test 
%
% fp -- Pointer to a file for the test output, defaults to 1
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

if nargin < 1
    fp = 1;
end

fname = mfilename();
fname = regexprep(fname, 'test_', '');


fprintf(fp, '================\n Test %s\n================\n', fname);

d = test_mpdcm_fmri_load_td();
[y, u, theta, ptheta] = mpdcm_fmri_tinput(d{1});

% Test whether there is any clear bug
try
    mpdcm_fmri_mle({y}, {u}, {theta}, ptheta);
    fprintf(fp, '       Passed\n');
catch err
    fprintf(fp, '   Not passed at line %d\n', err.stack(end).line);
    display(getReport(err, 'extended'));
end


end

