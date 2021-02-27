function tapas_BLT_initDatafile(params)
% tapas_BLT_initDatafile initializes the main data file for the BLT


% ________________________________________________________________________%
%
%                EVENTS (Time stamps for screen on and offsets)
% ________________________________________________________________________%

data.events.start_time          = GetSecs();
data.events.start_protocol      = [];
data.events.start_sequence      = [];

data.events.cue_pred_on         = [];
data.events.cue_pred_off        = [];

if params.pauseMode == 1
    data.events.pause_on            = [];
    data.events.pause_off           = [];
end

data.events.stim_on             = [];
data.events.stim_off            = [];

data.events.iti_on              = [];
data.events.iti_off             = [];

if params.stimAnsMode == 1
    data.events.stim_ans_on     = [];
    data.events.stim_ans_off    = [];
end

data.events.rating_on           = [];
data.events.rating_off          = [];

data.events.abort               = [];
data.events.end                 = [];


% ________________________________________________________________________%
%
%                             PREDICTIONS
% ________________________________________________________________________%

data.pred_answer                = [];
if params.predMode == 1
    data.pred_rt                = [];
end


% ________________________________________________________________________%
%
%                          RATING AND ANSWERS
% ________________________________________________________________________%

if params.stimAnsMode == 1
    data.stim_answer            = [];
    data.stim_answer_rt         = [];
end
data.rate_answer_diff           = [];
data.rate_answer_anx            = [];


save(params.path.datafile, 'data')