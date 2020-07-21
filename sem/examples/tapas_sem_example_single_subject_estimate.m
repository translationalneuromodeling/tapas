function [posterior] = tapas_sem_example_single_subject_estimate(model, param)
%% Example for estimate for single subjects.
%
% Input
%       model       -- String. Either seria or prosa
%       param       -- String. Parametric distribution.
% Output
%       posterior   -- Structure. Contains the posterior estimates.
%       summary     -- Table. Contains a table with a summary of the 
%                      posterior.

% aponteeduardo@gmail.com
% copyright (C) 2018
%


n = 0;

n = n + 1;
if nargin < n
    model = 'seria';
end

n = n + 1;
if nargin < n
    param = 'mixedgamma';
end

[data] = load_data();

switch model
case 'seria'
    ptheta = tapas_sem_seria_ptheta(); 
    switch param
        case 'invgamma'
            ptheta.llh = @c_seria_multi_invgamma;
        case 'gamma'
            ptheta.llh = @c_seria_multi_gamma;
        case 'mixedgamma'
            ptheta.llh = @c_seria_multi_mixedgamma;
        case 'lognorm'
            ptheta.llh = @c_seria_multi_lognorm;
        case 'later'
            ptheta.llh = @c_seria_multi_later;
        case 'wald'
            ptheta.llh = @c_seria_multi_wald;
        otherwise
            error('parametric function not defined')
    end

    ptheta.jm = [...
        eye(19)
        zeros(3, 8) eye(3) zeros(3, 8)];
    ptheta.p0(11) = tapas_logit([0.005], 1);
case 'prosa'
    ptheta = tapas_sem_prosa_ptheta(); % Choose at convinience.
    switch param
    case 'invgamma'
        ptheta.llh = @c_prosa_multi_invgamma;
    case 'gamma'
        ptheta.llh = @c_prosa_multi_gamma;
    case 'mixedgamma'
        ptheta.llh = @c_prosa_multi_mixedgamma;
    case 'lognorm'
        ptheta.llh = @c_prosa_multi_lognorm;
    case 'later'
        ptheta.llh = @c_prosa_multi_later;
    case 'wald'
        ptheta.llh = @c_prosa_multi_wald;
    otherwise
        error('parametric function not defined')
    end

    ptheta.jm = [...
        eye(15)
        zeros(3, 6) eye(3) zeros(3, 6)];

end

pars = struct();

pars.T = linspace(0.1, 1, 1).^5;
pars.nburnin = 10000;
pars.niter = 10000;
pars.ndiag = 500;
pars.mc3it = 4;
pars.verbose = 1;
pars.thinning = 100;

display(ptheta);
inference = struct();
inference.kernel_scale = 0.1 * 0.1;

posterior = cell(numel(data), 1);
for i = 1:numel(data)
    posterior = tapas_sem_single_subject_estimate(...
        data(i), ptheta, inference, pars);
    display(posterior);
    tapas_sem_display_posterior(posterior);
end

end

function [data] = load_data()

f = mfilename('fullpath');
[tdir, ~, ~] = fileparts(f);

data = load(fullfile(tdir, 'data', 'example_data.mat'));

data = data.data;

end
