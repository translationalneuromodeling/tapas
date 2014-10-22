function test_mpdcm_fmri_tinput()
%% Test of dcm_fmri_tinput.
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

display('===========================')
display('Testing mpdcm_fmri_tinput')
display('===========================')

d =  test_mpdcm_fmri_load_td();

[u, theta, ptheta] = mpdcm_fmri_tinput(d{1});

try
    mpdcm_fmri_int_check_input({u}, {theta}, ptheta);
    [y] = mpdcm_fmri_int({u}, {theta}, ptheta);
    display('   Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end
