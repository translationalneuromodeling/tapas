function [abpl, ubpl] = tapas_ti_bpl(llh, t)
%% Computes the Bayesiand predictive log likelihood.
%
%   Input
%       llh     Multidimensional array of preditive likelihood. Dimensions are
%               Number of subjects x number of chains x number of samples
%       t       Temperature schedule of size 1 x number of chains
%   Output
%       abpl    Adjusted Bayesian preditive likelihood
%       ubpl    Unadjusted Bayesian predictive likelihood
%

% aponteeduardo@gmail.com
% copyright (C) 2017
%

[ns, nc, np] = size(llh);
%if all([1, nc] == size(t))
%    error('tapas:ti:bpl', 'Dimensions of t and llh do not match');
%end

% Compute the free energy

lnz = trapz(t, sum(mean(llh, 3), 1));

% Make sure that the first dimension is the temperature
ubpl = trapz(t', mean(llh, 3)')';

% Do post hoc importance sampling
abpl = zeros(ns, 1);
for s = 1:ns
    sllh = bsxfun(@times, -t', squeeze(llh(s, :, :)));
    sllh = bsxfun(@minus, sllh, mean(sllh, 2));
    sllh = bsxfun(@minus, sllh, log(sum(exp(sllh), 2)));
    % Change the dynamic range to max being 100
    adja = -max(sllh')' + log(1000); 
    sllh = bsxfun(@plus, sllh, adja);
    sllh = exp(sllh) ;
    nllh = squeeze(sum(skip_column(llh, s), 1));
    nllh = sum(bsxfun(@times, nllh, sllh), 2) .* exp(-adja);
    abpl(s) = lnz - trapz(t, nllh);
end

end

function [nllh] = skip_column(llh, sbj)
% Removes an slice and return array minux slice

[ns, nc, np] = size(llh);
i = 1:ns;
i(sbj) = [];

nllh = llh(i, :, :);

end
