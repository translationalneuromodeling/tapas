function [ons_secs, outliersHigh, outliersLow] = tapas_physio_correct_cardiac_pulses_manually(ons_secs,thresh_cardiac)
% this function takes the onsets from ECG measure and controls for
% outliers (more or less than a threshold given by a percentile increased
% or decreased by upperTresh or lowerThresh percent respectively.
%
% Author: Jakob Heinzle, TNU,
%           adaptation: Lars Kasper
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_correct_cardiac_pulses_manually.m 469 2014-04-28 10:23:59Z kasperla $


percentile = thresh_cardiac.percentile;
upperThresh = thresh_cardiac.upper_thresh;
lowerThresh = thresh_cardiac.lower_thresh;

[outliersHigh,outliersLow,fh] = tapas_physio_cardiac_detect_outliers(ons_secs.cpulse, percentile, upperThresh, lowerThresh);
if any(outliersHigh)
    %disp('Press Enter to proceed to manual peak selection!');
    %pause;
    additionalPulse=[];
    fh2=figure;
    for outk=1:length(outliersHigh)
        s=0;
        while ~(s==1)
            indStart = outliersHigh(outk)-1; indEnd = outliersHigh(outk)+2;
            ind=find(ons_secs.t>=ons_secs.cpulse(indStart), 1, 'first')-100:find(ons_secs.t<=ons_secs.cpulse(indEnd), 1, 'last')+100;
            figure(fh2); clf;
            plot(ons_secs.t(ind),ons_secs.c(ind),'r')
            hold on;
            plot(ons_secs.cpulse(indStart:indEnd),ones(4,1)*max(ons_secs.c(ind)),'ok')
            inpNum=input('How many triggers do you want to set? Enter a number between 0 and 10 : ');
            I1=[];
            for ii=1:inpNum
                figure(fh2);
                [I1(ii), J1] = ginput(1);
                plot(I1(ii),J1, 'b*', 'MarkerSize',10);
                
            end
            
            s=input('If you agree with the selected triggers, press 1 (then enter) : ');
            if isempty(s)
                s=0;
            end
        end
        additionalPulse=[additionalPulse;I1'];
    end
    ons_secs.cpulse = sort([ons_secs.cpulse;additionalPulse]);
    close(fh2);
end
close(fh);


[outliersHigh,outliersLow,fh] = tapas_physio_cardiac_detect_outliers(ons_secs.cpulse, percentile, upperThresh, lowerThresh);

nPulses = length(ons_secs.cpulse);
finalIndex=1:nPulses;

nWindow = 10;

%show windows with suspicious outliers only
for indStart = round(nWindow/2):nWindow-2:nPulses
    indEnd = indStart+nWindow-1;
    if any(ismember(outliersLow, indStart:indEnd))
        %disp('Press Enter to proceed to manual peak deletion!');
        %pause;
        fh3=figure('Position', [500 500 1000 500]);
        ind=find(ons_secs.t>=ons_secs.cpulse(indStart), 1, 'first')-100:find(ons_secs.t<=ons_secs.cpulse(indEnd), 1, 'last')+100;
        figure(fh3); clf;
        plot(ons_secs.t(ind),ons_secs.c(ind),'r')
        hold on;
        plot(ons_secs.cpulse(indStart:indEnd),ones(nWindow,1)*max(ons_secs.c(ind)),'ko','MarkerFaceColor','r');
        alreadyDeleted=intersect(indStart:indEnd,setdiff(1:nPulses,finalIndex));
        plot(ons_secs.cpulse(alreadyDeleted),ones(size(alreadyDeleted))*max(ons_secs.c(ind)),'ro');
        for kk=indStart:indEnd
            text(ons_secs.cpulse(kk),max(ons_secs.c(ind))*1.05,int2str(kk));
        end
        
        delInd= [];
        
        delInd=input('Enter the indices of pulses you want to delete (0 if none, put multiple in [], e.g. [19,20]): ');
        
        if ~isempty(delInd) && any(find(delInd~=0))
            plot(ons_secs.cpulse(delInd),max(ons_secs.c(ind))*ones(size(delInd)), 'rx', 'MarkerSize',20);
            finalIndex=setdiff(finalIndex,delInd');
        end
        close(fh3);
    end
    
end
ons_secs.cpulse = sort(ons_secs.cpulse(finalIndex));
close(fh);
[outliersHigh,outliersLow,fh] = tapas_physio_cardiac_detect_outliers(ons_secs.cpulse, percentile, upperThresh, lowerThresh);
close(fh);
% recursively determine outliers
if ~isempty(outliersHigh) || ~isempty(outliersLow)
     doManualCorrectionAgain = input('More outliers detected after correction. Do you want to remove them? (1=yes, 0=no; ENTER)');
    if doManualCorrectionAgain
        [ons_secs, outliersHigh, outliersLow] = tapas_physio_correct_cardiac_pulses_manually(ons_secs,thresh_cardiac);
    end
end

cpulse = ons_secs.cpulse;
save(thresh_cardiac.file, 'ons_secs', 'thresh_cardiac', 'cpulse');
end
