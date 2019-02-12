function [funcs] = tapas_sem_get_model_functions(model, param)
%% Return the functions associated with a model.
%
% Input
%       model       -- String. Either seria or prosa
%       param       -- String. The name of a parametric distribution.
% Output
%       funcs       -- Struct. A structure of function handles.
%
%   funcs.model     ->  Name of the model
%   funcs.param     ->  Parametric distribution
%   funcs.summaries ->  Generate the summaries from the posterior 
%                       distributions


% aponteeduardo@gmail.com
% copyright (C) 2019
%

funcs = struct('model', model, 'param', param, 'summaries', []);

switch model
case 'seria'
    switch param
        case 'invgamma'
            funcs.summaries = @c_seria_summaries_invgamma;
        case 'gamma'
            funcs.summaries = @c_seria_summaries_gamma;
        case 'mixedgamma'
            funcs.summaries = @c_seria_summaries_mixedgamma;
        case 'lognorm'
            funcs.summaries = @c_seria_summaries_lognorm;
        case 'later'
            funcs.summaries = @c_seria_summaries_later;
        case 'wald'
            funcs.summaries = @c_seria_summaries_wald;
        otherwise
            error('tapas:sem:model_functions', 'Unknown parametric family')
    end
case 'prosa'
    switch param
        case 'invgamma'
            funcs.summaries = @c_prosa_summaries_invgamma;
        case 'gamma'
            funcs.summaries = @c_prosa_summaries_gamma;
        case 'mixedgamma'
            funcs.summaries = @c_prosa_summaries_mixedgamma;
        case 'lognorm'
            funcs.summaries = @c_prosa_summaries_lognorm;
        case 'later'
            funcs.summaries = @c_prosa_summaries_later;
        case 'wald'
            funcs.summaries = @c_prosa_summaries_wald;
        otherwise
            error('tapas:sem:model_functions', 'Unknown parametric family')
    end
otherwise
    error('tapas:sem:model_functions', 'Unknown model');
end


end
