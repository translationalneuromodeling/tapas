% Illustrates how posterior inferences can be visualized. See micp_demo_mcmc.m.

% Kay H. Brodersen, ETH Zurich, Switzerland
% $Id: micp_demo_mcmc_plot.m 16210 2012-05-31 07:04:48Z bkay $
% -------------------------------------------------------------------------
function micp_demo_mcmc_plot(ks,ns,samples_popu,samples_pijs,varargin)
    
    % Check input
    defaults.models = 1:size(samples_popu,1);
    defaults.names = [];
    defaults.xlabels = [];
    defaults.new_figure = true;
    defaults.axis_square = true;
    defaults.subplots = [2 4]; 
    defaults.i = 1; % first subplot to use
    defaults.show_data = true;
    defaults.show_distr = true;
    defaults.show_int = true;
    defaults.fontsize = 11;
    args = propval(varargin,defaults);
    
    % Initialization
    if args.new_figure, fh = figure; else fh = gcf; end
    set(fh,'name',mfilename,'color',[1 1 1]);
    i=args.i-1; letter=args.i-1;
    ks_bandwidth = []; %0.02;
    % Colors: 1:grey 2:green 3:black 4:red 5:blue 6:light-grey
    colors = [140 140 140; 0 138 82; 0 0 0; 214 32 32; 0 112 192; 220 220 220]/255;
    m = size(ns,2);
    
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
    % Show subject-specific posteriors
    if ~isempty(samples_pijs)
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        plot([-100 100],[0.5 0.5],'--','color',[0.5 0.5 0.5],'linewidth',2);
        [~,I] = sort(accs);
        % sample accuracies
        h1 = plot(accs(I),'.','color',colors(5,:),'markersize',15);
        % beta-binomial model: subject-specific posteriors
        mo=1;
        x = 1:m;
        ss = squeeze(samples_pijs(mo,:,:));
        y = mean(ss,2)';
        lowerlength = []; upperlength = [];
        for j=1:m
            lowerlength(j) = y(j)-percentile(ss(j,:),2.5);
            upperlength(j) = percentile(ss(j,:),97.5)-y(j);
        end
        y = y(I); lowerlength = lowerlength(I); upperlength = upperlength(I);
        h2 = plot(x,y,'ok');
        errorbar(x,y,lowerlength,upperlength,'linestyle','none','marker','none','linewidth',1,'color',[0 0 0]);
        %
        letter=letter+1; title([figLetter(letter),'subject-specific inference'],'fontsize',args.fontsize);
        set(gca,'box','on','fontsize',args.fontsize);
        xlabel('subjects (sorted)');
        ylabel('balanced accuracy'); % Requires that samples_pijs are baccs!
        axis([0 m+1 0 1]);
        if args.axis_square, axis square; end
        h=legend([h1 h2],'sample accuracies','posterior means','location','southeast');
        set(h,'fontsize',args.fontsize-2);
    end
    
    % ---------------------------------------------------------------------
    % Plot comparison of population-mean intervals
    if args.show_int && ~isempty(samples_popu)
        i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
        x=0;
        ha=0;
        % chance bar
        plot([-100 100],[0.5 0.5],'--','color',[0.5 0.5 0.5],'linewidth',2);
        for mo=args.models
            ss = samples_popu(mo,:);
            y = mean(ss);
            lowerlength = y-percentile(ss,2.5);
            upperlength = percentile(ss,97.5)-y;
            x=x+1; ha=ha+1; h(ha) = plot(x,y,'ok');
            errorbar(x,y,lowerlength,upperlength,'LineWidth',2,'color',[0 0 0]);
        end
        % conventional mean sample accuracy and 95% CI
        t = tinv(1-0.05/2,m-1);
        sem = std(accs)/sqrt(m);
        y = mean(accs);
        lowerlength = t*sem;
        upperlength = t*sem;
        x=x+1; ha=ha+1; h(ha) = plot(x,y,'o','color',[192 0 0]/255);
        errorbar(x,y,lowerlength,upperlength,'linewidth',2,'color',[192 0 0]/255);
        % finalise figure
        axis([0 x+1 0 1]);
        set(gca,'box','on','xtick',[],'fontsize',args.fontsize);
        letter=letter+1; title([figLetter(letter),'population-mean intervals'],'fontsize',args.fontsize);
        if args.axis_square, axis square; end
        tmp_names = args.names; tmp_names{end+1} = 'classical confidence interval';
        h=legend(h,tmp_names,'location','southeast');
        set(h,'fontsize',args.fontsize-2);
    end
    
    % ---------------------------------------------------------------------
    % For each model, plot posterior distribution of population mean
    if args.show_distr && ~isempty(samples_popu)
        v4 = 0;
        for mo=args.models
            i=i+1;subplot(args.subplots(1),args.subplots(2),i);hold on;
            ss = samples_popu(mo,:);
            [y,x] = ksdensity(ss,[0:0.001:1],'width',ks_bandwidth);
            plot(x,y,'k-');
            inner = (percentile(ss,5) <= x);
            area(x(inner),y(inner),'facecolor',colors(6,:));
            plotChanceLevel;
            letter=letter+1; title([figLetter(letter),args.names{mo}],'fontsize',args.fontsize);
            set(gca,'box','on','fontsize',args.fontsize);
            if ~isempty(args.xlabels), xlabel(['population mean ', args.xlabels{mo}]); end
            v = axis;
            v4 = max([v(4) v4]);
            if args.axis_square, axis square; end
        end
        
        % Finalise plots: print infraliminal probability
        i=i-length(args.models);
        for mo=args.models
            i=i+1;subplot(args.subplots(1),args.subplots(2),i);
            v=axis; v(4) = v4; axis(v);
            ss = samples_popu(mo,:); % population samples
            textXP = ['p(<0.5|k) = ', sprintf('%0.3f',sum(ss<0.5)/length(ss))];
            text(0.05,0.95*v4,textXP,'verticalalignment','top','backgroundcolor',[1 1 1]);
        end
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
