function [cpulse, verbose] = tapas_physio_get_cardiac_pulses(t, c, thresh_cardiac, cardiac_modality, verbose)
% extract heartbeat events from ECG or pulse oximetry time course
%
%   cpulse = tapas_physio_get_cardiac_pulses(t, c, thresh_cardiac, cardiac_modality, verbose);
%
% IN
%   t                  vector of time series of log file (in seconds, corresponds to c)
%   c                  raw time series of ECG or pulse oximeter
%   thresh_cardiac      is a structure with the following elements
%           .modality - 'ecg' or 'oxy'; ECG or Pulse oximeter used?
%           .min -     - for modality 'ECG': [percent peak height of sample QRS wave]
%                      if set, ECG heartbeat event is calculated from ECG
%                      timeseries by detecting local maxima of
%                      cross-correlation to a sample QRS-wave;
%                      leave empty, to use Philips' log of heartbeat event
%                      - for modality 'OXY': [peak height of pulse oxymeter] if set, pulse
%                      oxymeter data is used and minimal peak height
%                      set is used to determined maxima
%           .kRpeakfile
%                    - [default: not set] string of file containing a
%                      reference ECG-peak
%                      if set, ECG peak detection via cross-correlation (via
%                      setting .ECG_min) performed with a saved reference ECG peak
%                      This file is saved after picking the QRS-wave
%                      manually (i.e. if .ECG_min is set), so that
%                      results are reproducible
%           .manual_peak_select [false] or true; if true, a user input is
%           required to specify a characteristic R-peak interval in the ECG
%           or pulse oximetry time series
%   verbose            debugging plot for thresholding, only provided, if verbose.level >=2
%
% OUT
%
% EXAMPLE
%   ons_samples.cpulse = tapas_physio_get_cardiac_pulses(ons_secs.c,
%   thresh.cardiac, cardiac_modality, cardiac_peak_file);
%
%   See also tapas_physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2013-02-16
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_cardiac_pulses.m 235 2013-08-19 16:28:07Z kasperla $

%% detection of cardiac R-peaks

% debug=true;
debug=false;
switch lower(cardiac_modality)
case 'oxy_old'
     c = c-mean(c); c = c./max(c); % normalize time series
        dt = t(2)-t(1);
        dt120 = round(0.5/dt); % heart rate < 120 bpm
        
        % smooth noisy pulse oximetry data to detect peaks
        w = gausswin(dt120,1);
        sc = conv(c, w, 'same');
        sc = sc-mean(sc); sc = sc./max(sc); % normalize time series

        % Highpass filter to remove drifts
        cutoff=1/dt; %1 seconds/per sampling units
        forder=2;
        [b,a]=butter(forder,2/cutoff, 'high');
        sc =filter(b,a, sc);
        sc = sc./max(sc);

                [tmp, cpulse] = tapas_physio_findpeaks(sc,'minpeakheight',thresh_cardiac.min,'minpeakdistance', dt120); 
        
        if verbose.level >=2 % visualise influence of smoothing on peak detection
            verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
            set(gcf, 'Name', 'PPU-OXY: Tresholding Maxima for Heart Beat Detection');
            [tmp, cpulse2] = tapas_physio_findpeaks(c,'minpeakheight',thresh_cardiac.min,'minpeakdistance', dt120);
            plot(t, c, 'k');
            hold all;
            plot(t, sc, 'r', 'LineWidth', 2);
            hold all
            hold all;stem(t(cpulse2),c(cpulse2), 'k--');
            hold all;stem(t(cpulse),sc(cpulse), 'm', 'LineWidth', 2);
            plot(t, repmat(thresh_cardiac.min, length(t),1),'g-');
            legend('Raw PPU time series', 'Smoothed PPU Time Series', ...
            'Detected Heartbeats in Raw Time Series', ...
            'Detected Heartbeats in Smoothed Time Series', ...
            'Threshold (Min) for Heartbeat Detection');
        end
        
        cpulse = t(cpulse);

        
        %courtesy of Steffen Bollmann, KiSpi Zurich
        case 'oxy'
        c = c-mean(c); c = c./std(c); % normalize time series
        dt = t(2)-t(1);
        dt120 = round(0.5/dt); % heart rate < 120 bpm
        
        
        % DEBUG
        if debug
            figure(1);clf;
            subplot 211;
            plot(t, c, 'k');
            hold all;
        end
        % DEBUG
        
        %guess peaks in two steps with updated avereage heartrate
        %first step
        [tmp, cpulseFirstGuess] = tapas_physio_findpeaks( ...
            c,'minpeakheight',thresh_cardiac.min,'minpeakdistance', dt120);
        
        %second step
        averageHeartRateInSamples = round(mean(diff(cpulseFirstGuess)));
        [tmp, cpulseSecondGuess] = tapas_physio_findpeaks(c,...
            'minpeakheight',thresh_cardiac.min,...
            'minpeakdistance', round(0.5*averageHeartRateInSamples));
        
        if debug
%             hold on;
%             stem(t(cpulseSecondGuess),4*ones(length(cpulseSecondGuess),1),'r')
        end
        
        %test signal/detection quality
        signalQualityIsBad = false;
        
        distanceBetweenPeaks = diff(cpulseSecondGuess);
        
        
        nBins = length(distanceBetweenPeaks)/10;
        [n,dtbin] = hist(distanceBetweenPeaks,nBins);
        
        percentile = 0.8;
        deviationPercent = 60;
        iPercentile = find(cumsum(n)>percentile*sum(n),1,'first');
        if dtbin(end) > (1+deviationPercent/100)*dtbin(iPercentile)
            signalQualityIsBad = true;
        end
        
        %always use the correlation based method for testing
%         signalQualityIsBad = true;
       
        if signalQualityIsBad
            %build template based on the guessed peaks
            halfTemplateWidthInSeconds = 0.2;
            halfTempalteWithInSamples = halfTemplateWidthInSeconds / dt;
            for n=2:numel(cpulseSecondGuess)-2
                startTemplate = cpulseSecondGuess(n)-halfTempalteWithInSamples;
                endTemplate = cpulseSecondGuess(n)+halfTempalteWithInSamples;
                
                template(n,:) = c(startTemplate:endTemplate);
            end
            
            %delete first zero-elements of the template
            template(1,:) = [];
            
            pulseTemplate = mean(template);
            
            % delete the peaks deviating from the mean too
            % much before building the final template
            for n=1:size(template,1)
                correlation = corrcoef(template(n,:),pulseTemplate);
                similarityToTemplate(n) = correlation(1,2);
            end
            
            count = 1;
            for n=1:size(template,1)
                if similarityToTemplate(n) > 0.95
                    cleanedTemplate(count,:)=template(n,:);
                    count=count+1;
                end
            end
            
            if exist('cleanedTemplate','var')
                pulseCleanedTemplate=mean(cleanedTemplate);
                clear similarityToTemplate
            else
                pulseCleanedTemplate = mean(template);
            end
            
            %determine starting peak for the search
            forStart=2*halfTempalteWithInSamples+1;
            forEnd=cpulseSecondGuess(20);
            for n=forStart:forEnd
                startSignalIndex=n-halfTempalteWithInSamples;
                endSignalIndex=n+halfTempalteWithInSamples;
                
                signalPart = c(startSignalIndex:endSignalIndex);
                correlation = corrcoef(signalPart,pulseCleanedTemplate);
                
                %Debug
                %             figure(2);clf;
                %             plot(signalPart);
                %             hold all;
                %             plot(pulseCleanedTemplate);
                %Debug
                
                similarityToTemplate(n) = correlation(1,2);
            end
            
            [C_bestMatch,I_bestMatch] = max(similarityToTemplate);
            clear similarityToTemplate
            
            %now compute backwards to the beginning
            n=I_bestMatch;
            peakNumber = 1;
            similarityToTemplate=zeros(size(t),1);
            
            searchStepsTotal=round(0.5*averageHeartRateInSamples);
            while n > 1+searchStepsTotal+halfTempalteWithInSamples
                for searchPosition=-searchStepsTotal:1:searchStepsTotal
                    startSignalIndex=n-halfTempalteWithInSamples+searchPosition;
                    endSignalIndex=n+halfTempalteWithInSamples+searchPosition;
                    
                    signalPart = c(startSignalIndex:endSignalIndex);
                    correlation = corrcoef(signalPart,pulseCleanedTemplate);
                    
                    %DEBUG
%                     if debug
%                         figure(1);
%                         subplot 212;
%                         plot(signalPart);
%                         hold all;
%                         plot(pulseCleanedTemplate);
%                         hold off;
%                     end
                    %DEBUG
                    
                    gaussianWindow = gausswin(2*searchStepsTotal+1);
%                     currentWeight = gaussianWindow(searchPosition+searchStepsTotal+1);

                    currentWeight = abs(c(n+searchPosition+1));
                    correlationWeighted =  currentWeight .* correlation(1,2);
                    similarityToTemplate(n+searchPosition) = correlationWeighted;
                    
                end
                
                %DEBUG
                if debug
                    if (n>100) && (n< size(c,1)-100)
                        figure(1);subplot 211;plot(t,similarityToTemplate,'b-')
                        xlim([t(n-100) t(n+100)])
                    end
                end
                %DEBUG
                
                
                %find biggest correlation-peak from the last search
                indexSearchStart=n-searchStepsTotal;
                indexSearchEnd=n+searchStepsTotal;
                
                indexSearchRange=indexSearchStart:indexSearchEnd;
                searchRangeValues=similarityToTemplate(indexSearchRange);
                [C_bestMatch,I_bestMatch] = max(searchRangeValues);
                bestPosition = indexSearchRange(I_bestMatch);
                
                cpulse(peakNumber) = bestPosition;
                peakNumber = peakNumber+1;
                

                n=bestPosition-averageHeartRateInSamples;
            end

            
            n=bestPosition;
            peakNumber=1;
            clear cpulse;
            %now correlate template with PPU signal at the positions
            %where we would expect a peak based on the average heartrate and
            %search in the neighborhood for the best peak, but weight the peaks
            %deviating from the initial starting point by a gaussian
            searchStepsTotal=round(0.5*averageHeartRateInSamples);
            
            if n< searchStepsTotal+halfTempalteWithInSamples+1
                n=searchStepsTotal+halfTempalteWithInSamples+1;
            end
            
            while n < size(c,1)-searchStepsTotal-halfTempalteWithInSamples
                %search around peak
                
                for searchPosition=-searchStepsTotal:1:searchStepsTotal
                    startSignalIndex=n-halfTempalteWithInSamples+searchPosition;
                    endSignalIndex=n+halfTempalteWithInSamples+searchPosition;
                    
                    signalPart = c(startSignalIndex:endSignalIndex);
                    correlation = corrcoef(signalPart,pulseCleanedTemplate);
                    
                    %DEBUG
%                     if debug
%                         figure(1);
%                         subplot 212;
%                         plot(signalPart);
%                         hold all;
%                         plot(pulseCleanedTemplate);
%                         hold off;
%                     end
                    %DEBUG
                    
                    gaussianWindow = gausswin(2*searchStepsTotal+1);
                    locationWeight = gaussianWindow(searchPosition+searchStepsTotal+1);
%                     locationWeight = 1;
                    amplitudeWeight = abs(c(n+searchPosition+1));
%                     amplitudeWeight = 1;
                    correlationWeighted =  locationWeight .* amplitudeWeight .* correlation(1,2);
                    similarityToTemplate(n+searchPosition) = correlationWeighted;
                    
                    
                end
                
                %DEBUG
                if debug
                    if (n>100) && (n< size(c,1)-100)
                        figure(1);subplot 211;plot(t,similarityToTemplate,'b-')
                        xlim([t(n-100) t(n+100)])
                    end
                end
                %DEBUG
                
                
                %find biggest correlation-peak from the last search
                indexSearchStart=n-searchStepsTotal;
                indexSearchEnd=n+searchStepsTotal;
                
                indexSearchRange=indexSearchStart:indexSearchEnd;
                searchRangeValues=similarityToTemplate(indexSearchRange);
                [C_bestMatch,I_bestMatch] = max(searchRangeValues);
                bestPosition = indexSearchRange(I_bestMatch);
                
                if debug
                    stem(t(bestPosition),4,'g');
                end
                
                cpulse(peakNumber) = bestPosition;
                peakNumber = peakNumber+1;
                
                %only take the last 20 cpulses to compute the current HeartRate
                foundCpulses = size(cpulse,2);
                
                if  foundCpulses < 3
                    currentHeartRateInSamples=averageHeartRateInSamples;
                end
                
                if  (foundCpulses < 21) && (foundCpulses >= 3)
                    currentHeartRateInSamples = round(mean(diff(cpulse)));
                end
                
                if foundCpulses > 20
                    currentCpulses = cpulse (foundCpulses-20:foundCpulses);
                    currentHeartRateInSamples = round(mean(diff(currentCpulses)));
                end
                
                
                %check currentHeartRate
                checkSmaller=currentHeartRateInSamples > 0.5*averageHeartRateInSamples;
                checkLarger=currentHeartRateInSamples < 1.5*averageHeartRateInSamples;
                
                %jumpToNextPeakSearchArea
                if (checkSmaller && checkLarger)
                    n=bestPosition+currentHeartRateInSamples;
                else
                    n=bestPosition+averageHeartRateInSamples;
                end
            end
        end
        
        if ~signalQualityIsBad
            cpulse = cpulseSecondGuess;
        end
        
        
        if verbose.level >=2
            verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
            titstr = 'PPU-OXY: Heart Beat Detection';
            set(gcf, 'Name', titstr);           
            plot(t, c, 'k');
            hold all;
            stem(t(cpulse),4*ones(size(cpulse)), 'r');
            legend('PPU time course', 'Detected cardiac pulses');           
            title(titstr);
        end
        
        
        cpulse = t(cpulse);
        
        
        
    case 'ecg'
        do_manual_peakfind = strcmp(thresh_cardiac.method, 'manual');
        if do_manual_peakfind
            thresh_cardiac.kRpeak = [];
        else
            ECGfile = load(thresh_cardiac.file);
            thresh_cardiac.min = ECGfile.ECG_min;
            thresh_cardiac.kRpeak = ECGfile.kRpeak;
        end
        
        inp_events = [];
        ECG_min = thresh_cardiac.min;
        kRpeak = thresh_cardiac.kRpeak;
        if do_manual_peakfind
            while ECG_min
                [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ECG_min, [], inp_events);
                ECG_min = input('Press 0, then return, if right ECG peaks were found, otherwise type next numerical choice for ECG_min and continue the selection: ');
            end
        else
            [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ECG_min, kRpeak, inp_events);
        end
        ECG_min = ECG_min_new;
        cpulse = t(cpulse);
        % save manually found peak parameters to file
        if do_manual_peakfind
            save(thresh_cardiac.file, 'ECG_min', 'kRpeak');
        end
    otherwise
        disp('How did you measure your cardiac cycle, dude? (ECG, OXY)');
end
