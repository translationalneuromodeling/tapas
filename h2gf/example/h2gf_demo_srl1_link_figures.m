function h2gf_demo_srl1_link_figures(NrIter,config_file,h2gf_parameter)

%% specify which configuration of the binary hgf has been used
if config_file == 1
    configtype = 'estka2';
elseif config_file == 2
    configtype = 'estka2mu2';
elseif config_file == 3
    configtype = 'estka2mu3';
elseif config_file == 4
    configtype = 'estka2om3';
elseif config_file == 5
    configtype = 'estka2sa2';
elseif config_file == 6
    configtype = 'estka2sa3';
elseif config_file == 7
    configtype = 'estom2';
elseif config_file == 8
    configtype = 'estom2mu2';
elseif config_file == 9
    configtype = 'estom2mu3';
elseif config_file == 10
    configtype = 'estom2om3';
elseif config_file == 11
    configtype = 'estom2sa2';
elseif config_file == 12
    configtype = 'estom2sa3';
end

%% specify which h2gf parameter should be plotted
if h2gf_parameter == 1
    parameter_label = 'LME';
    parameter_title = 'LME';
elseif h2gf_parameter == 2
    parameter_label = 'ka';
    parameter_title = 'ka';
elseif h2gf_parameter == 3
    parameter_label = 'om2';
    parameter_title = 'om2';
elseif h2gf_parameter == 4
    parameter_label = 'om3';
    parameter_title = 'om3';
elseif h2gf_parameter == 5
    parameter_label = 'mu2_0';
    parameter_title = 'mu2 0';
elseif h2gf_parameter == 6
    parameter_label = 'mu3_0';
    parameter_title = 'mu3 0';
elseif h2gf_parameter == 7
    parameter_label = 'sa2_0';
    parameter_title = 'sa2 0';
elseif h2gf_parameter == 8
    parameter_label = 'sa3_0';
    parameter_title = 'sa3 0';
end

%% define where results have been stored:
% f = mfilename('fullpath');
%
% [tdir, ~, ~] = fileparts(f);
tdir = '/cluster/project/tnu/igsandra/tapas/h2gf/example/';
% tdir = 'D:\PRSSI\h2gf\';
for eta_label =1:6
    try
    if eta_label == 1
        maskResFolder = ([tdir,'/results/',configtype,'/eta', num2str(eta_label),'/', num2str(NrIter)]);
        fig1=hgload([maskResFolder,['/srl1_h2gf_',parameter_label,'_boxplot_',configtype,'_eta',num2str(eta_label),'_', num2str(NrIter),'.fig']]);
        fig1.CurrentAxes.XTickLabel=[1:5:41];
    elseif eta_label == 2
        maskResFolder = ([tdir,'/results/',configtype,'/eta', num2str(eta_label),'/', num2str(NrIter)]);   
        fig2=hgload([maskResFolder,['/srl1_h2gf_',parameter_label,'_boxplot_',configtype,'_eta',num2str(eta_label),'_', num2str(NrIter),'.fig']]);
        fig2.CurrentAxes.XTickLabel=[1:5:41];
    elseif eta_label == 3
        maskResFolder = ([tdir,'/results/',configtype,'/eta', num2str(eta_label),'/', num2str(NrIter)]);  
        fig3=hgload([maskResFolder,['/srl1_h2gf_',parameter_label,'_boxplot_',configtype,'_eta',num2str(eta_label),'_', num2str(NrIter),'.fig']]);
        fig3.CurrentAxes.XTickLabel=[1:5:41];
    elseif eta_label == 4
        maskResFolder = ([tdir,'/results/',configtype,'/eta', num2str(eta_label),'/', num2str(NrIter)]);  
        fig4=hgload([maskResFolder,['/srl1_h2gf_',parameter_label,'_boxplot_',configtype,'_eta',num2str(eta_label),'_', num2str(NrIter),'.fig']]);
        fig4.CurrentAxes.XTickLabel=[1:5:41];
    elseif eta_label == 5
        maskResFolder = ([tdir,'/results/',configtype,'/eta', num2str(eta_label),'/', num2str(NrIter)]); 
        fig5=hgload([maskResFolder,['/srl1_h2gf_',parameter_label,'_boxplot_',configtype,'_eta',num2str(eta_label),'_', num2str(NrIter),'.fig']]);
        fig5.CurrentAxes.XTickLabel=[1:5:41];
    elseif eta_label == 6
        maskResFolder = ([tdir,'/results/',configtype,'/eta', num2str(eta_label),'/', num2str(NrIter)]);
        fig6=hgload([maskResFolder,['/srl1_h2gf_',parameter_label,'_boxplot_',configtype,'_eta',num2str(eta_label),'_', num2str(NrIter),'.fig']]);
        fig6.CurrentAxes.XTickLabel=[1:5:41];
    end
    catch e
        fprintf(1,'There was an error! The message was:\n%s',e.message);
        close all;
    end
    
end
% 2) Prepare subplots figure
fig7=figure('rend','painters','pos',[10 10 1300 700],'Name',['srl1; NrIter: ',num2str(NrIter),'; Parameter; ',parameter_label, '; Config: ',configtype],'NumberTitle','off')
h(1)=subplot(3,2,1);
h(2)=subplot(3,2,2);
h(3)=subplot(3,2,3);
h(4)=subplot(3,2,4);
h(5)=subplot(3,2,5);
h(6)=subplot(3,2,6);
% 3) Paste figures on the subplots

copyobj(allchild(get(fig1,'CurrentAxes')),h(1));
copyobj(allchild(get(fig2,'CurrentAxes')),h(2));

copyobj(allchild(get(fig3,'CurrentAxes')),h(3));
copyobj(allchild(get(fig4,'CurrentAxes')),h(4));

copyobj(allchild(get(fig5,'CurrentAxes')),h(5));
copyobj(allchild(get(fig6,'CurrentAxes')),h(6));
% % 4) Add legends to the new plot % %
l(1)=legend(h(1),'eta = 1');
l(2)=legend(h(2),'eta = 10');
l(3)=legend(h(3),'eta = 20');
l(4)=legend(h(4),'eta = 40');
l(5)=legend(h(5),'eta om3 = 5');
l(6)=legend(h(6),'eta om3 = 10');

linkaxes([h(1),h(2),h(3),h(4),h(5),h(6)],'x')
h(1).XLim = [1,40];
YLim1 = h(1).YLim;
h(1).YLim = YLim1;

suptitle([parameter_title, '; ',configtype, '; nr. samples: ',num2str(NrIter)]);
fig7.Color = [1 1 1];

cd([tdir,'/results'])
saveas(fig7,[configtype,'_srl1_NrIter_',num2str(NrIter),'parameter',parameter_label, '_config',], 'fig');
print(fig7,[configtype,'_srl1_NrIter_',num2str(NrIter),'parameter',parameter_label, '_config.png'], '-dpng','-r300');
print(fig7,'-dpdf', [configtype,'_srl1_NrIter_',num2str(NrIter),'parameter',parameter_label, '_config.pdf']);
end

