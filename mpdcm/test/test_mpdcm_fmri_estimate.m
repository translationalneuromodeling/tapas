function test_mpdcm_fmri_estimate()
%% 
%
% aponteeduardo@gmail.com
% copyright (C) 2014
%

d = test_mpdcm_fmri_load_td();

% Check that there are no fatal error

nd = d{1};
nd.U.u = nd.U.u(1:4:end,:);
nd.U.dt = 1.0;

try
    pars = struct();
    pars.T = linspace(1e-1, 1, 100).^5;
    pars.nburnin = 100;
    pars.niter = 100;
    dcm = mpdcm_fmri_estimate(nd, pars);   
    display('    Passed')
catch err
    d = dbstack();
    fprintf('   Not passed at line %d\n', d(1).line)
    disp(getReport(err, 'extended'));
end

% Make a more intensive test of the function
for i = 1:5
    try
        pars = struct();
        pars.T = linspace(1e-1, 1, 100).^5;
        pars.nburnin = 50;
        pars.niter = 50;

        tic
        dcm = mpdcm_fmri_estimate(d{1}, pars);
        toc

        fprintf('fe: %0.5f\n', dcm.F);
        display('    Passed')
    catch err
        d = dbstack();
        fprintf('   Not passed at line %d\n', d(i).line)
        disp(getReport(err, 'extended'));
    end
end
end
