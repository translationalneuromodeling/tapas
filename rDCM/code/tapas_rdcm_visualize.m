function tapas_rdcm_visualize(output, DCM, options, plot_regions, plot_mode)
% tapas_rdcm_visualize(output, DCM, options, plot_regions, plot_mode)
% 
% Generates a simple graphical output of the rDCM results. 
% 
%   Input:
%   	output          - model inversion results
%   	DCM             - model structure
%       options         - estimation options
%       plot_regions    - array of region indices
%       plot_mode       - plot power spectral density (of temporal derivative) 
%                         or BOLD signal time series
%
%   Output: 
%
 
% ----------------------------------------------------------------------
% 
% Authors: Stefan Fraessle (stefanf@biomed.ee.ethz.ch), Ekaterina I. Lomakina
% 
% Copyright (C) 2016-2022 Translational Neuromodeling Unit
%                         Institute for Biomedical Engineering
%                         University of Zurich & ETH Zurich
%
% This file is part of the TAPAS rDCM Toolbox, which is released under the 
% terms of the GNU General Public License (GPL), version 3.0 or later. You
% can redistribute and/or modify the code under the terms of the GPL. For
% further see COPYING or <http://www.gnu.org/licenses/>.
% 
% Please note that this toolbox is in an early stage of development. Changes 
% are likely to occur in future releases.
% 
% ----------------------------------------------------------------------


% default if no plot mode is specified
if ( nargin < 5 || isempty(plot_mode) )
    plot_mode = 1;
end

% default if no regions are specified
if ( nargin < 4 || isempty(plot_regions) )
    plot_regions = 1;
end

% specify a colormap for plotting the adjacency matrices
% (if the "cbrewer" tool is in your MATLAB path, a "pretty" colormap will
%  be selected from the cbrewer-library, otherwise the default MATLAB
%  colormap is used)
try
    [cmap] = cbrewer('div', 'RdBu', 71, 'PCHIP');
    cmap   = flipud(cmap);
catch
    cmap   = 'parula';
end


% test of predicted and actual power spectral density or BOLD signal time 
% series should be plotted
if ( plot_mode == 1 )
    if ( isfield(output.signal,'yd_source_fft') && isfield(output.signal,'yd_pred_rdcm_fft') )
        plotSignal = 1;
    else
        plotSignal = 0;
    end
elseif ( plot_mode == 2 )
    if ( isfield(output.signal,'y_source') && isfield(output.signal,'y_pred_rdcm') )
        plotSignal = 1;
    else
        plotSignal = 0;
    end
end



%% visualize results

% simulation (where true parameters are known) or empirical analysis
if ( options.visualize )
    if ( options.type == 's' )
        if ( plotSignal == 1 )

            % visualize estimated connectivity pattern
            figure('units','normalized','outerposition',[0 0 1 1])
            sub1 = subplot(2,2,1);
            colormap(sub1,cmap)
            imagesc(output.Ep.A)
            title('estimated','FontSize',14)
            axis square
            caxis([-1*max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A))))) max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A)))))])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',12)
            ylabel('region (to)','FontSize',12)

            % visualize true connectivity pattern
            sub2 = subplot(2,2,2);
            colormap(sub2,cmap)
            imagesc(DCM.Tp.A)
            title('true','FontSize',14)
            axis square
            caxis([-1*max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A))))) max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A)))))])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',12)
            ylabel('region (to)','FontSize',12)

            % get the samples to plot
            if ( plot_mode == 1 )
                y_source_reshape    = reshape(output.signal.yd_source_fft,length(output.signal.yd_source_fft)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_pred_rdcm_reshape = reshape(output.signal.yd_pred_rdcm_fft,length(output.signal.yd_pred_rdcm_fft)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_source_reshape    = abs(y_source_reshape(1:size(y_source_reshape,1)/2,plot_regions)).^2;
                y_pred_rdcm_reshape	= abs(y_pred_rdcm_reshape(1:size(y_pred_rdcm_reshape,1)/2,plot_regions)).^2;
            elseif ( plot_mode == 2 )
                y_source_reshape    = reshape(output.signal.y_source,length(output.signal.y_source)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_pred_rdcm_reshape = reshape(output.signal.y_pred_rdcm,length(output.signal.y_source)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_source_reshape    = y_source_reshape(:,plot_regions);
                y_pred_rdcm_reshape	= y_pred_rdcm_reshape(:,plot_regions);
            end
                
            % visualize true and predicted BOLD signal
            subplot(2,1,2)
            hold on
            ha(1) = plot(y_source_reshape(:),'Color',[0.7002 0.7088 0.7004],'LineWidth',1.5);
            ha(2) = plot(y_pred_rdcm_reshape(:),'Color',[0.5059 0.0118 0.1255],'LineWidth',1.5);
            yl = ylim;
            for int = 1:size(y_pred_rdcm_reshape,2)-1, plot([int*size(y_pred_rdcm_reshape,1) int*size(y_pred_rdcm_reshape,1)],yl,'k.-'), end
            xlim([0 numel(y_source_reshape)])
            set(gca,'xtick',[1 size(y_source_reshape,1)/2 size(y_source_reshape,1)])
            set(gca,'xticklabel',[1 size(y_source_reshape,1)/2 size(y_source_reshape,1)])
            legend(ha,{'true','predicted'},'Location','NE','FontSize',12)
            legend boxoff
            xlabel('sample index','FontSize',12)
            if ( plot_mode == 1 )
                if ( numel(plot_regions) == 1 ), axis square, end
                title('true and prediced power spectral density','FontSize',14);
                ylabel('PSD','FontSize',12)
            elseif ( plot_mode == 2 )
                title('true and prediced time series','FontSize',14);
                ylabel('BOLD','FontSize',12)
            end
            
        else

            % visualize estimated connectivity pattern
            figure('units','normalized','outerposition',[0 0 1 1])
            sub1 = subplot(1,2,1);
            colormap(sub1,cmap)
            imagesc(output.Ep.A)
            title('estimated','FontSize',16)
            axis square
            caxis([-1*max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A))))) max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A)))))])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',14)
            ylabel('region (to)','FontSize',14)

            % visualize true connectivity pattern
            sub2 = subplot(1,2,2);
            colormap(sub2,cmap)
            imagesc(DCM.Tp.A)
            title('true','FontSize',16)
            axis square
            caxis([-1*max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A))))) max(max(abs(DCM.Tp.A-diag(diag(DCM.Tp.A)))))])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',14)
            ylabel('region (to)','FontSize',14)

        end

    else

        if ( plotSignal == 1 )

            % visualize estimated connectivity pattern
            figure('units','normalized','outerposition',[0 0 1 1])
            sub1 = subplot(2,2,1);
            colormap(sub1,cmap)
            imagesc(output.Ep.A)
            title('estimated','FontSize',14)
            axis square
            caxis([-1*max(max(abs(output.Ep.A-diag(diag(output.Ep.A))))) max(max(abs(output.Ep.A-diag(diag(output.Ep.A)))))])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',12)
            ylabel('region (to)','FontSize',12)

            % visualize posterior probability of binary indicator variable
            sub2 = subplot(2,2,2);
            colormap(sub2,'gray')
            imagesc(output.Ip.A)
            title('Pp binary indicator','FontSize',14)
            axis square
            caxis([0 1])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',12)
            ylabel('region (to)','FontSize',12)

            
            % get the samples to plot
            if ( plot_mode == 1 )
                y_source_reshape    = reshape(output.signal.yd_source_fft,length(output.signal.yd_source_fft)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_pred_rdcm_reshape = reshape(output.signal.yd_pred_rdcm_fft,length(output.signal.yd_pred_rdcm_fft)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_source_reshape    = abs(y_source_reshape(1:size(y_source_reshape,1)/2,plot_regions)).^2;
                y_pred_rdcm_reshape	= abs(y_pred_rdcm_reshape(1:size(y_pred_rdcm_reshape,1)/2,plot_regions)).^2;
            elseif ( plot_mode == 2 )
                y_source_reshape    = reshape(output.signal.y_source,length(output.signal.y_source)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_pred_rdcm_reshape = reshape(output.signal.y_pred_rdcm,length(output.signal.y_source)/size(output.Ep.A,1),size(output.Ep.A,1));
                y_source_reshape    = y_source_reshape(:,plot_regions);
                y_pred_rdcm_reshape	= y_pred_rdcm_reshape(:,plot_regions);
            end
            
            % visualize measured and predicted BOLD signal
            subplot(2,1,2);
            hold on
            ha(1) = plot(y_source_reshape(:),'Color',[0.7002 0.7088 0.7004],'LineWidth',1.5);
            ha(2) = plot(y_pred_rdcm_reshape(:),'Color',[0.5059 0.0118 0.1255],'LineWidth',1.5);
            yl = ylim;
            for int = 1:size(y_pred_rdcm_reshape,2)-1, plot([int*size(y_pred_rdcm_reshape,1) int*size(y_pred_rdcm_reshape,1)],yl,'k.-'), end
            xlim([0 numel(y_source_reshape)])
            set(gca,'xtick',[1 size(y_source_reshape,1)/2 size(y_source_reshape,1)])
            set(gca,'xticklabel',[1 size(y_source_reshape,1)/2 size(y_source_reshape,1)])
            legend(ha,{'true','predicted'},'Location','NE','FontSize',12)
            legend boxoff
            xlabel('sample index','FontSize',12)
            if ( plot_mode == 1 )
                if ( numel(plot_regions) == 1 ), axis square, end
                title('true and prediced power spectral density','FontSize',14);
                ylabel('PSD','FontSize',12)
            elseif ( plot_mode == 2 )
                title('true and prediced time series','FontSize',14);
                ylabel('BOLD','FontSize',12)
            end

        else

            % visualize estimated connectivity pattern
            figure('units','normalized','outerposition',[0 0 1 1])
            sub1 = subplot(1,2,1);
            colormap(sub1,cmap)
            imagesc(output.Ep.A)
            title('estimated','FontSize',16)
            axis square
            caxis([-1*max(max(abs(output.Ep.A-diag(diag(output.Ep.A))))) max(max(abs(output.Ep.A-diag(diag(output.Ep.A)))))])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',14)
            ylabel('region (to)','FontSize',14)

            % visualize posterior probability of binary indicator variable
            sub2 = subplot(1,2,2);
            colormap(sub2,'gray')
            imagesc(output.Ip.A)
            title('Pp binary indicator','FontSize',16)
            axis square
            caxis([0 1])
            colorbar
            set(gca,'xtick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            set(gca,'ytick',[1 round(size(output.Ep.A,1)/2) size(output.Ep.A,1)])
            xlabel('region (from)','FontSize',14)
            ylabel('region (to)','FontSize',14)

        end
    end
end

end
