function [v,raw_params] = create_read_param_struct(file)

dotind = findstr(file,'.');
ending = lower(file(dotind(end)+1:end));

switch ending
    case 'rec'
        parfile = [file(1:dotind),'par'];
        par = parread(parfile);
        v.slice = unique(par.ImageInformation.SliceNumber);
        v.echo = unique(par.ImageInformation.EchoNumber);
        v.dyn = unique(par.ImageInformation.DynamicScanNumber);  
        v.phase =  unique(par.ImageInformation.CardiacPhaseNumber);
        v.typ = unique(par.ImageInformation.ImageTypeMr);
        raw_params = par;
     case 'par'
        par = parread(file);
        v.slice = unique(par.ImageInformation.SliceNumber);
        v.echo = unique(par.ImageInformation.EchoNumber);
        v.dyn = unique(par.ImageInformation.DynamicScanNumber);  
        v.phase =  unique(par.ImageInformation.CardiacPhaseNumber);
        v.typ = unique(par.ImageInformation.ImageTypeMr);
        raw_params = par;
    case 'cpx'
        header = read_cpx_header(file,'no');
        v.loca = unique(header(:,1))+1;
        v.slice = unique(header(:,2))+1;
        v.coil = unique(header(:,3))+1;
        v.phase = unique(header(:,4))+1;
        v.echo = unique(header(:,5))+1;
        v.dyn = unique(header(:,6))+1;
        v.seg = unique(header(:,7))+1;        
        v.seg2 = unique(header(:,18))+1;
        raw_params = header;
    case 'data'
        t = 'TEHROA';
        listfile = [file(1:dotind),'list'];
        list = listread(listfile);
        typ = unique(list.Index.typ(:,2));
        for i = 1:length(typ)
            numtyp(i) = findstr(typ(i),t);
        end
        v.typ = sort(numtyp);
        v.mix = unique(list.Index.mix)+1;
        v.dyn = unique(list.Index.dyn)+1;
        v.phase = unique(list.Index.card)+1;
        v.echo = unique(list.Index.echo)+1;
        v.loca = unique(list.Index.loca)+1;
        v.coil = unique(list.Index.chan)+1;
        v.seg = unique(list.Index.extr1)+1;
        v.seg2 = unique(list.Index.extr2)+1;
        v.ky = unique(list.Index.ky)+1;
        v.slice = unique(list.Index.kz)+1;
        v.aver = unique(list.Index.aver)+1;
        raw_params = list;
    case 'list'        
        t = 'TEHROA';
        list = listread(file);
        typ = unique(list.Index.typ(:,2));
        for i = 1:length(typ)
            numtyp(i) = findstr(typ(i),t);
        end
        v.typ = sort(numtyp);
        v.mix = unique(list.Index.mix)+1;
        v.dyn = unique(list.Index.dyn)+1;
        v.phase = unique(list.Index.card)+1;
        v.echo = unique(list.Index.echo)+1;
        v.loca = unique(list.Index.loca)+1;
        v.coil = unique(list.Index.chan)+1;
        v.seg = unique(list.Index.extr1)+1;
        v.seg2 = unique(list.Index.extr2)+1;
        v.ky = unique(list.Index.ky)+1;
        v.slice = unique(list.Index.kz)+1;
        v.aver = unique(list.Index.aver)+1; 
        raw_params = list;
    otherwise
        v = -1;
end
            