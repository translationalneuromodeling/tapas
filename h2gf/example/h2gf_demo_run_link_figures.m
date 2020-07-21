%% link figures based on which configuration of the binary hgf has been used
addpath(genpath('/cluster/project/tnu/igsandra/tapas/'));
%% h2gf parameter that will be plotted
% if h2gf_parameter == 1
%     parameter_label = 'LME';
% elseif h2gf_parameter == 2
%     parameter_label = 'ka';
% elseif h2gf_parameter == 3
%     parameter_label = 'om2';
% elseif h2gf_parameter == 4
%     parameter_label = 'om3';
% elseif h2gf_parameter == 5
%     parameter_label = 'mu2_0';
% elseif h2gf_parameter == 6
%     parameter_label = 'mu3_0';
% elseif h2gf_parameter == 7
%     parameter_label = 'sa2_0';
% elseif h2gf_parameter == 8
%     parameter_label = 'sa3_0';
% end

% config_file == 1
% configtype = 'estka2';
for c = 1:2
h2gf_demo_srl2_link_figures(1000, 1, c);close all
h2gf_demo_srl2_link_figures(3000, 1, c);close all
h2gf_demo_srl2_link_figures(4000, 1, c);close all
h2gf_demo_srl2_link_figures(5000, 1, c);close all
h2gf_demo_srl1_link_figures(1000, 1, c);close all
h2gf_demo_srl2_link_figures(3000, 1, c);close all
h2gf_demo_srl3_link_figures(4000, 1, c);close all
h2gf_demo_srl4_link_figures(5000, 1, c);close all
end


% config_file == 2
% configtype = 'estka2mu2';
for c = [1 2 5]
h2gf_demo_srl2_link_figures(1000, 2, c);close all
h2gf_demo_srl2_link_figures(3000, 2, c);close all
h2gf_demo_srl2_link_figures(4000, 2, c);close all
h2gf_demo_srl2_link_figures(5000, 2, c);close all
h2gf_demo_srl1_link_figures(1000, 2, c);close all
h2gf_demo_srl2_link_figures(3000, 2, c);close all
h2gf_demo_srl3_link_figures(4000, 2, c);close all
h2gf_demo_srl4_link_figures(5000, 2, c);close all
end


% config_file == 3
% configtype = 'estka2mu3';
for c = [1 2 6]
h2gf_demo_srl2_link_figures(1000, 3, c);close all
h2gf_demo_srl2_link_figures(3000, 3, c);close all
h2gf_demo_srl2_link_figures(4000, 3, c);close all
h2gf_demo_srl2_link_figures(5000, 3, c);close all
h2gf_demo_srl1_link_figures(1000, 3, c);close all
h2gf_demo_srl1_link_figures(3000, 3, c);close all
h2gf_demo_srl1_link_figures(4000, 3, c);close all
h2gf_demo_srl1_link_figures(5000, 3, c);close all
end


% config_file == 4
% configtype = 'estka2om3';
for c = [1 2 4]
h2gf_demo_srl2_link_figures(1000, 4, c);close all
h2gf_demo_srl2_link_figures(3000, 4, c);close all
h2gf_demo_srl2_link_figures(4000, 4, c);close all
h2gf_demo_srl2_link_figures(5000, 4, c);close all
h2gf_demo_srl1_link_figures(1000, 4, c);close all
h2gf_demo_srl1_link_figures(3000, 4, c);close all
h2gf_demo_srl1_link_figures(4000, 4, c);close all
h2gf_demo_srl1_link_figures(5000, 4, c);close all
end


% config_file == 5
% configtype = 'estka2sa2';
for c = [1 2 7]
h2gf_demo_srl2_link_figures(1000, 5, c);close all
h2gf_demo_srl2_link_figures(3000, 5, c);close all
h2gf_demo_srl2_link_figures(4000, 5, c);close all
h2gf_demo_srl2_link_figures(5000, 5, c);close all
h2gf_demo_srl1_link_figures(1000, 5, c);close all
h2gf_demo_srl1_link_figures(3000, 5, c);close all
h2gf_demo_srl1_link_figures(4000, 5, c);close all
h2gf_demo_srl1_link_figures(5000, 5, c);close all
end


% config_file == 6
% configtype = 'estka2sa3';
for c = [1 2 8]
h2gf_demo_srl2_link_figures(1000, 6, c);close all
h2gf_demo_srl2_link_figures(3000, 6, c);close all
h2gf_demo_srl2_link_figures(4000, 6, c);close all
h2gf_demo_srl2_link_figures(5000, 6, c);close all
h2gf_demo_srl1_link_figures(1000, 6, c);close all
h2gf_demo_srl1_link_figures(3000, 6, c);close all
h2gf_demo_srl1_link_figures(4000, 6, c);close all
h2gf_demo_srl1_link_figures(5000, 6, c);close all
end

% config_file == 7
% configtype = 'estom2';
for c = [1 3]
h2gf_demo_srl2_link_figures(1000, 7, c);close all
h2gf_demo_srl2_link_figures(3000, 7, c);close all
h2gf_demo_srl2_link_figures(4000, 7, c);close all
h2gf_demo_srl2_link_figures(5000, 7, c);close all
h2gf_demo_srl1_link_figures(1000, 7, c);close all
h2gf_demo_srl1_link_figures(3000, 7, c);close all
h2gf_demo_srl1_link_figures(4000, 7, c);close all
h2gf_demo_srl1_link_figures(5000, 7, c);close all
end


% config_file == 8
% configtype = 'estom2mu2';
for c = [1 3 5]
h2gf_demo_srl2_link_figures(1000, 8, c);close all
h2gf_demo_srl2_link_figures(3000, 8, c);close all
h2gf_demo_srl2_link_figures(4000, 8, c);close all
h2gf_demo_srl2_link_figures(5000, 8, c);close all
h2gf_demo_srl1_link_figures(1000, 8, c);close all
h2gf_demo_srl1_link_figures(3000, 8, c);close all
h2gf_demo_srl1_link_figures(4000, 8, c);close all
h2gf_demo_srl1_link_figures(5000, 8, c);close all
end


% config_file == 9
% configtype = 'estom2mu3';
for c = [1 3 6]
h2gf_demo_srl2_link_figures(1000, 9, c);close all
h2gf_demo_srl2_link_figures(3000, 9, c);close all
h2gf_demo_srl2_link_figures(4000, 9, c);close all
h2gf_demo_srl2_link_figures(5000, 9, c);close all
h2gf_demo_srl1_link_figures(1000, 9, c);close all
h2gf_demo_srl1_link_figures(3000, 9, c);close all
h2gf_demo_srl1_link_figures(4000, 9, c);close all
h2gf_demo_srl1_link_figures(5000, 9, c);close all
end


% config_file == 10
% configtype = 'estom2om3';
for c = [1 3 4]
h2gf_demo_srl2_link_figures(1000, 10, c);close all
h2gf_demo_srl2_link_figures(3000, 10, c);close all
h2gf_demo_srl2_link_figures(4000, 10, c);close all
h2gf_demo_srl2_link_figures(5000, 10, c);close all
h2gf_demo_srl1_link_figures(1000, 10, c);close all
h2gf_demo_srl1_link_figures(3000, 10, c);close all
h2gf_demo_srl1_link_figures(4000, 10, c);close all
h2gf_demo_srl1_link_figures(5000, 10, c);close all
end


% config_file == 11
% configtype = 'estom2sa2';
for c = [1 3 7]
h2gf_demo_srl2_link_figures(1000, 11, c);close all
h2gf_demo_srl2_link_figures(3000, 11, c);close all
h2gf_demo_srl2_link_figures(4000, 11, c);close all
h2gf_demo_srl2_link_figures(5000, 11, c);close all
h2gf_demo_srl1_link_figures(1000, 11, c);close all
h2gf_demo_srl1_link_figures(3000, 11, c);close all
h2gf_demo_srl1_link_figures(4000, 11, c);close all
h2gf_demo_srl1_link_figures(5000, 11, c);close all
end


% config_file == 12
% configtype = 'estom2sa3';
for c = [1 3 8]
h2gf_demo_srl2_link_figures(1000, 12, c);close all
h2gf_demo_srl2_link_figures(3000, 12, c);close all
h2gf_demo_srl2_link_figures(4000, 12, c);close all
h2gf_demo_srl2_link_figures(5000, 12, c);close all
h2gf_demo_srl1_link_figures(1000, 12, c);close all
h2gf_demo_srl1_link_figures(3000, 12, c);close all
h2gf_demo_srl1_link_figures(4000, 12, c);close all
h2gf_demo_srl1_link_figures(5000, 12, c);close all
end