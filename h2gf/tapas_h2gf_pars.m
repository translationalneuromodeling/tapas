function [pars] = tapas_h2gf_pars(data, model, pars)
%% Set up default parameters of the estimator. 
%
% Input
%       data         --
%       model       -- 
%       pars        --
% Output
%       pars        -- Updated parameters
%       

% aponteeduardo@gmail.com
% copyright (C) 2019
%

if ~isfield('pars', 'model_evidence_method')
    pars.model_evidence_method = 'wbic';
end

switch lower(pars.model_evidence_method)
case 'wbic'
    if isfield(pars, 'T')
    warning('tapas:h2gf:wbic', ...
        'Using WBIC: temperature schedule will be ignored');
    end
    ns = numel(data);
    T = ones(ns, 2);
    assert(ns > 3, ...
        'WBIC is only valid when the number of subjects is more than 3');
    T(:, 1) = 1/log(ns);
    pars.T = T; 
case 'ti'
    % pass
otherwise
    error('tapas:h2gf:input', ...
        'model_evidence_method %s not supported', pars.model_evidence_method);
end

pars = tapas_hgf_pars(data, model, pars);

end
