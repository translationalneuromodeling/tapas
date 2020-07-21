function fh = tapas_physio_plot_raw_physdata_siemens_tics(tCardiac, c, tRespiration, r, ...
    hasCardiacFile, hasRespirationFile, cpulse, rpulse, cacq_codes, ...
    racq_codes)
% Plots raw physiological data from Siemens TICS format
%
%   output = tapas_physio_plot_raw_physdata_siemens_tics(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_plot_raw_physdata_siemens_tics
%
%   See also

% Author: Lars Kasper
% Created: 2017-09-10
% Copyright (C) 2017 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%

fh = tapas_physio_get_default_fig_params();
stringTitle = 'Read-In: Siemens Tics - Read-in cardiac and respiratory logfiles';
set(gcf, 'Name', stringTitle);
stringLegend = {};
tOffset = min([tRespiration; tCardiac]);
hp = [];
if hasCardiacFile
    
    ampl = max(abs(c));
    if ~isempty(cacq_codes)
        
        acq_codes = cacq_codes;
        % create different stem plots for different acq-codes => are summed
        % up, so look at binary, powers of 2
        acq_codes = num2str(dec2bin(acq_codes));
        
        % add 4 letter 1st acq code to make access operations posssible
        acq_codes = char('0000', acq_codes);
        acq_codes(1,:) = [];
        iSampleECGTrigger = [];
        iSampleOxyTrigger = [];
        iSampleExtTrigger = [];
           
        if ~isempty(acq_codes)
            iSampleECGTrigger = find(acq_codes(:,end)=='1'); % 1
            iSampleOxyTrigger = find(acq_codes(:,end-1)=='1'); % 2
            iSampleExtTrigger = find(acq_codes(:,end-3)=='1'); % 8
        end
        
        colors = {'r', 'm', 'k'};
        iSamples = {iSampleECGTrigger, iSampleOxyTrigger, iSampleExtTrigger};
        strStemLegend = {'ECG Pulse Trigger (Siemens detect)', ...
            'Oxy Pulse Trigger (Siemens detect)', 'External Scanner Trigger'};
        
        for s = numel(iSamples):-1:1
            iSample = iSamples{s};
            if ~isempty(iSample)
                hp(end+1) = stem(tCardiac(iSample)-tOffset, ...
                    ampl*(1+s/10)*ones(size(acq_codes(iSample))));
                hold all;
                set(hp(end), 'Color', colors{s});
                stringLegend{1,end+1} = strStemLegend{s};
            end
            
        end
        
    end
    
    hold all;
    
    if ~isempty(cpulse)
        hp(end+1) = stem(cpulse-tOffset, ampl*ones(size(cpulse)));
        set(hp(end), 'Color', 'r');
        stringLegend{1, end+1} = 'Detected hearbeats';
    end
    
    
    hp(end+1) = plot(tCardiac-tOffset, c, 'r.-'); hold all;
    stringLegend{1, end+1} =  ...
        sprintf('Cardiac time course, start time %5.2e', tOffset);
    
    
    
end

if hasRespirationFile
    
    ampl = max(abs(r));
    if ~isempty(racq_codes)
        
        
        acq_codes = racq_codes;
        % create different stem plots for different acq-codes => are summed
        % up, so look at binary, powers of 2
        acq_codes = num2str(dec2bin(acq_codes));
        
       
        % add 4 letter 1st acq code to make access operations posssible
        acq_codes = char('0000', acq_codes);
        acq_codes(1,:) = [];
        iSampleRespTrigger = [];
        iSampleExtTrigger = [];
           
        if ~isempty(acq_codes)
            iSampleRespTrigger = find(acq_codes(:,end-2)=='1'); % 4
            iSampleExtTrigger = find(acq_codes(:,end-3)=='1'); % 8
        end
        
        colors = {'g', 'k'};
        iSamples = {iSampleRespTrigger, iSampleExtTrigger};
        strStemLegend = {'Resp Pulse Trigger (Siemens detect)', ...
            'External Scanner Trigger'};
        
        for s = numel(iSamples):-1:1
            iSample = iSamples{s};
            if ~isempty(iSample)
                hp(end+1) = stem(tRespiration(iSample)-tOffset, ...
                    ampl*(1+s/10)*ones(size(acq_codes(iSample))));
                set(hp(end), 'Color', colors{s});
                stringLegend{1,end+1} = strStemLegend{s};
            end
            
        end
        
    end
    
    hold all;
    
    if ~isempty(rpulse)
        hp(end+1) = stem(rpulse-tOffset, ampl*ones(size(rpulse)));
        set(hp(end), 'Color', [0 0.8 0]);
        stringLegend{1, end+1} = 'Detected Breath starts';
    end
    
    hp(end+1) = plot(tRespiration-tOffset, r, 'g.-');
    stringLegend{1, end+1} =  ...
        sprintf('Respiratory time course, start time %5.2e', tOffset);
    
end
xlabel('t (seconds)');
legend(handle(hp), stringLegend);
title(stringTitle);
end