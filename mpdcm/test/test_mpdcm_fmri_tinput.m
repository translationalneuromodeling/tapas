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

P = cell(numel(d), 1);
U = cell(numel(d), 1);

for i = 1:numel(d)
    P{i} = d{i}.M.pE;
    U{i} = d{i}.U;
end

[u, theta, ptheta] = mpdcm_fmri_tinput(U, P);

try
    mpdcm_fmri_int_check_input(u, theta, ptheta);
    [y] = mpdcm_fmri_int(u, theta, ptheta);
    display('   Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
end
