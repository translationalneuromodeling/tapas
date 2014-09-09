% Illustrates how posterior inferences can be visualized. See micp_demo_vb.m.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: micp_demo_vb_plot.m 16210 2012-05-31 07:04:48Z bkay $
% -------------------------------------------------------------------------
function micp_demo_vb_plot(data,infc,varargin)
    
    % Check input
    defaults.new_figure = true;
    defaults.axis_square = true;
    defaults.subplots = [2 3]; 
    defaults.i = 1; % first subplot to use
    defaults.show_data = true;
    defaults.show_pop = true;
    defaults.show_intervals = true;
    defaults.show_subj = true;
    defaults.fontsize = 11;
    args = propval(varargin,defaults);
    
    % Unpack data
    ks = data.ks;
    ns = data.ns;
    m = size(ns,2);
    
    % Initialization
    if args.new_figure, fh = figure; else fh = gcf; end
    set(fh,'name',mfilename,'color',[1 1 1]);
    i=args.i-1; letter=args.i-1;
    % Colors: 1:grey 2:green 3:black 4:red 5:blue 6:light-grey
    colors = [140 140 140; 0 138 82; 0 0 0; 214 32 32; 0 112 192; 220 220 220]/255;

    % ---------------------------------------------------------------------
    % Plot number of correct predictions
    if args.show_data
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        bar([1:length(ns)]-0.2, ns(1,:), 0.4,'facecolor',colors(6,:),'edgecolor','none');
        bar([1:length(ns)]-0.2, ks(1,:), 0.4,'facecolor',colors(2,:),'edgecolor','none');
        bar([1:length(ns)]+0.2, ns(2,:), 0.4,'facecolor',colors(6,:),'edgecolor','none');
        bar([1:length(ns)]+0.2, ks(2,:), 0.4,'facecolor',colors(4,:),'edgecolor','none');
        axis([1-0.6 m+0.6 0 max(max(ns))]);
        set(gca,'box','on','fontsize',args.fontsize);
        xlabel('subjects');
        ylabel('correct predictions');
        letter=letter+1; title([figLetter(letter),'data'],'fontsize',args.fontsize);
        if args.axis_square, axis square; end
    end
    
    % ---------------------------------------------------------------------
    % Plot pos. vs. neg. sample accuracies
    if args.show_data
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        tpr = ks(1,:)./ns(1,:);
        tnr = ks(2,:)./ns(2,:);
        plot(tnr,tpr,'.','color',colors(5,:),'markersize',15)
        set(gca,'box','on','fontsize',args.fontsize);
        xlabel('true negative rate');
        ylabel('true positive rate');
        axis([0 1 0 1]);
        letter=letter+1; title([figLetter(letter),'data'],'fontsize',args.fontsize);
        if args.axis_square, axis square; end
    end
    
    % ---------------------------------------------------------------------
    % Sample accuracies
    accs = sum(ks,1)./sum(ns,1);
    baccs = 0.5*(ks(1,:)./ns(1,:) + ks(2,:)./ns(2,:));
    
    % ---------------------------------------------------------------------
    % Plot posterior distribution of population mean
    if args.show_pop && ~isempty(infc)
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        qp = infc.qp;
        qn = infc.qn;
        x = [0:0.001:1];
        y = logitnavgpdf(x,qp.mu_mu,sqrt(1/qp.eta_mu),qn.mu_mu,sqrt(1/qn.eta_mu));
        plot(x,y,'-','color',colors(5,:));
        if args.axis_square, axis square; end
        plotChanceLevel;
        xlabel(['population mean'],'fontsize',args.fontsize);
        letter=letter+1; title([figLetter(letter),'population inference'], ...
            'fontsize',args.fontsize);
        ylabel('balanced accuracy');
        set(gca,'box','on');
        set(gca,'fontsize',args.fontsize);
    end
    
    % ---------------------------------------------------------------------
    % Plot population-mean interval
    if args.show_intervals && ~isempty(infc)
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        x=0;
        % chance bar
        plot([-100 100],[0.5 0.5],'--','color',[0.5 0.5 0.5]);
        % posterior population mean and interval
        qp = infc.qp;
        qn = infc.qn;
        y = logitnavgmean(qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu));
        lowerlength = y-logitnavginv(0.025,qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu));
        upperlength = logitnavginv(0.975,qp.mu_mu,1/sqrt(qp.eta_mu),qn.mu_mu,1/sqrt(qn.eta_mu))-y;
        x=x+1; plot(x,y,'.','markersize',10,'color',colors(5,:));
        errorbar(x,y,lowerlength,upperlength,'LineWidth',2,'color',colors(5,:));
        % finalise
        axis([0 x+1 0 1]);
        if args.axis_square, axis square; end
        set(gca,'box','on','xtick',[]);
        letter=letter+1; title([figLetter(letter),'population inference'],...
            'fontsize',args.fontsize);
        set(gca,'fontsize',args.fontsize);
        ylabel('balanced accuracy','fontsize',args.fontsize);
    end
    
    % ---------------------------------------------------------------------
    % Show subject-specific posteriors
    if args.show_subj && ~isempty(infc)
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        plot([-100 100],[0.5 0.5],'--','color',[0.5 0.5 0.5],'linewidth',2);
        [~,I] = sort(baccs);
        % sample accuracies
        handle1 = plot(baccs(I),'.','color',colors(5,:),'markersize',15);
        % subject-specific posteriors (tnb)
        mo=1;
        qp = infc(mo).qp;
        qn = infc(mo).qn;
        x = 1:m;
        y = logitnavgmean(qp.mu_rho,1./sqrt(qp.eta_rho),qn.mu_rho,1./sqrt(qn.eta_rho));
        lowerlength = y - logitnavginv(0.025,qp.mu_rho,1./sqrt(qp.eta_rho),qn.mu_rho,1./sqrt(qn.eta_rho));
        upperlength = logitnavginv(0.975,qp.mu_rho,1./sqrt(qp.eta_rho),qn.mu_rho,1./sqrt(qn.eta_rho)) - y;
        y = y(I); lowerlength = lowerlength(I); upperlength = upperlength(I);
        handle2 = plot(x,y,'ok');
        errorbar(x,y,lowerlength,upperlength,'linestyle','none','marker','none','linewidth',2,'color',[0 0 0]);
        %
        letter=letter+1; title([figLetter(letter),'subject inference'],'fontsize',args.fontsize);
        set(gca,'box','on');
        xlabel('subjects (sorted)','fontsize',args.fontsize);
        ylabel('balanced accuracy','fontsize',args.fontsize);
        axis([0 m+1 0 1]);
        if args.axis_square, axis square; end
        set(gca,'fontsize',args.fontsize);
        h = legend([handle1,handle2],'sample balanced acc.','posterior mean','location','southeast');
    end

end

% -------------------------------------------------------------------------
function plotChanceLevel
    v = axis;
    plot([0.5 0.5], [0 1000], '--', 'color', [.5 .5 .5], 'linewidth', 2);
    axis(v);
end

% -------------------------------------------------------------------------
function str = figLetter(i)
    str = ['(',char(96+i),')  '];
end
