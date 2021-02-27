function [ text ] = tapas_BLT_initVisuals(params)
% tapas_BLT_initVisuals initializes Textures for all task relevant images
%
% Inputs:
%   params: data storage file
%
% Outputs:
%	text: image textures
%

% Set prediction cue images according to experiment mode and cue type
switch params.expMode
    case {'train_4', 'train_6'}
        if params.cuetype ==  1 
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue3_predYL_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue3_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue3_predYL_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue3_predYL_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue4_predYL_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue4_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue4_predYL_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue4_predYL_selR_img.png');
        elseif params.cuetype ==  2
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue3_predYR_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue3_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue3_predYR_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue3_predYR_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue4_predYR_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue4_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue4_predYR_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue4_predYR_selR_img.png');
        elseif params.cuetype ==  3 
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue4_predYL_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue4_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue4_predYL_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue4_predYL_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue3_predYL_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue3_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue3_predYL_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue3_predYL_selR_img.png');
        elseif params.cuetype ==  4
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue4_predYR_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue4_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue4_predYR_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue4_predYR_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue3_predYR_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue3_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue3_predYR_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue3_predYR_selR_img.png');
        end
    case {'task', 'debug'}
        if params.cuetype ==  1 
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue1_predYL_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue1_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue1_predYL_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue1_predYL_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue2_predYL_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue2_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue2_predYL_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue2_predYL_selR_img.png');
        elseif params.cuetype ==  2
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue1_predYR_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue1_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue1_predYR_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue1_predYR_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue2_predYR_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue2_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue2_predYR_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue2_predYR_selR_img.png');
        elseif params.cuetype ==  3 
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue2_predYL_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue2_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue2_predYL_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue2_predYL_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue1_predYL_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue1_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue1_predYL_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue1_predYL_selR_img.png');
        elseif params.cuetype ==  4
            params.stimuli.cue1         = fullfile(params.path.imgfolder,'cue2_predYR_img.png');
            params.stimuli.cue1Scale    = fullfile(params.path.imgfolder,'cue2_predScale.png');
            params.stimuli.cue1SL       = fullfile(params.path.imgfolder,'cue2_predYR_selL_img.png');
            params.stimuli.cue1SR       = fullfile(params.path.imgfolder,'cue2_predYR_selR_img.png');
            params.stimuli.cue2         = fullfile(params.path.imgfolder,'cue1_predYR_img.png');
            params.stimuli.cue2Scale    = fullfile(params.path.imgfolder,'cue1_predScale.png');
            params.stimuli.cue2SL       = fullfile(params.path.imgfolder,'cue1_predYR_selL_img.png');
            params.stimuli.cue2SR       = fullfile(params.path.imgfolder,'cue1_predYR_selR_img.png');
        end     
end

% Set stimulus answer images according to cue type
if params.cuetype ==  1 || params.cuetype ==  3
    params.stimuli.stimAnswer           = fullfile(params.path.imgfolder,'stimAnsYL_img.png');
    params.stimuli.stimAnswerSL         = fullfile(params.path.imgfolder,'stimAnsYL_selL_img.png');
    params.stimuli.stimAnswerSR         = fullfile(params.path.imgfolder,'stimAnsYL_selR_img.png');
elseif params.cuetype == 2 || params.cuetype ==  4
    params.stimuli.stimAnswer           = fullfile(params.path.imgfolder,'stimAnsYR_img.png');
    params.stimuli.stimAnswerSL         = fullfile(params.path.imgfolder,'stimAnsYR_selL_img.png');
    params.stimuli.stimAnswerSR         = fullfile(params.path.imgfolder,'stimAnsYR_selR_img.png');
end

% Add extra images   
params.stimuli.predtimeout      = fullfile(params.path.imgfolder,'pred_timeout_img.png');
params.stimuli.prednoresponse   = fullfile(params.path.imgfolder,'pred_noResponse.png');
params.stimuli.stimulation      = fullfile(params.path.imgfolder,'stimulation.png');
params.stimuli.iti              = fullfile(params.path.imgfolder,'iti.png');
params.stimuli.break            = fullfile(params.path.imgfolder,'break.png');
params.stimuli.ratetimeout      = fullfile(params.path.imgfolder,'rate_timeout.png');
params.stimuli.endfieldmap      = fullfile(params.path.imgfolder,'endfieldmap.png');
params.stimuli.endfinal         = fullfile(params.path.imgfolder,'endfinal.png');


for i = 1:params.nIntroTrain
    params.stimuli.introTrain{i}  = fullfile(params.path.imgfolder,['Slide',num2str(i),'.png']);
    imgIntroTrain = imread(params.stimuli.introTrain{i});
    text.IntroTrain{i} = Screen('MakeTexture', params.screen.window, imgIntroTrain);
end

for i2 = 1:params.nIntroMain
    params.stimuli.introMain{i2}  = fullfile(params.path.imgfolder,['Slide',num2str(i+i2),'.png']);
    imgIntroMain = imread(params.stimuli.introMain{i2});
    text.IntroMain{i2} = Screen('MakeTexture', params.screen.window, imgIntroMain);
end

% read images
imgcue1                         = imread(params.stimuli.cue1);
imgcue2                         = imread(params.stimuli.cue2);
imgcue1Scale                    = imread(params.stimuli.cue1Scale);
imgcue2Scale                    = imread(params.stimuli.cue2Scale);
imgcue1SL                       = imread(params.stimuli.cue1SL);
imgcue1SR                       = imread(params.stimuli.cue1SR);
imgcue2SL                       = imread(params.stimuli.cue2SL);
imgcue2SR                       = imread(params.stimuli.cue2SR);
imgpredtimeout                  = imread(params.stimuli.predtimeout);
imgprednoresponse               = imread(params.stimuli.prednoresponse);
imgstimulation                  = imread(params.stimuli.stimulation);
imgstimanswer                   = imread(params.stimuli.stimAnswer);
imgstimanswerSL                 = imread(params.stimuli.stimAnswerSL);
imgstimanswerSR                 = imread(params.stimuli.stimAnswerSR);
imgiti                          = imread(params.stimuli.iti);
imgbreak                        = imread(params.stimuli.break);
imgratetimeout                  = imread(params.stimuli.ratetimeout);
imgendfieldmap                  = imread(params.stimuli.endfieldmap);
imgendfinal                     = imread(params.stimuli.endfinal);

% Make the images into a textures that can be drawn to the screen
text.imgcue1              = Screen('MakeTexture', params.screen.window, imgcue1);
text.imgcue2              = Screen('MakeTexture', params.screen.window, imgcue2);
text.imgcue1Scale         = Screen('MakeTexture', params.screen.window, imgcue1Scale);
text.imgcue2Scale         = Screen('MakeTexture', params.screen.window, imgcue2Scale);
text.imgcue1SL            = Screen('MakeTexture', params.screen.window, imgcue1SL);
text.imgcue1SR            = Screen('MakeTexture', params.screen.window, imgcue1SR);
text.imgcue2SL            = Screen('MakeTexture', params.screen.window, imgcue2SL);
text.imgcue2SR            = Screen('MakeTexture', params.screen.window, imgcue2SR);
text.imgpredtimeout       = Screen('MakeTexture', params.screen.window, imgpredtimeout);
text.imgprednoresponse    = Screen('MakeTexture', params.screen.window, imgprednoresponse);
text.imgstimulation       = Screen('MakeTexture', params.screen.window, imgstimulation);
text.imgstimanswer        = Screen('MakeTexture', params.screen.window, imgstimanswer);
text.imgstimanswerSL      = Screen('MakeTexture', params.screen.window, imgstimanswerSL);
text.imgstimanswerSR      = Screen('MakeTexture', params.screen.window, imgstimanswerSR);
text.imgiti               = Screen('MakeTexture', params.screen.window, imgiti);
text.imgbreak             = Screen('MakeTexture', params.screen.window, imgbreak);
text.imgratetimeout       = Screen('MakeTexture', params.screen.window, imgratetimeout);
text.endfieldmap          = Screen('MakeTexture', params.screen.window, imgendfieldmap);
text.endfinal             = Screen('MakeTexture', params.screen.window, imgendfinal);
end

