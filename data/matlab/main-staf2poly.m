clc;
close all;
clear all;

cd 'D:/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

fileID = fopen('rmse-staf.txt','w');

for D = 3:3
    fprintf(fileID, 'd = %d\n', D);
    %for s = [09 20 22 23 24 25 27 29 30 31 32 35 37 39 40 41 42 43 44 45]
    %for s = [27 29 30 31 32 35 37 39 40 41 42 43 44 45]
    for s = [21]
        fprintf('sample %d\n', s);
        output_dir = ['data-staf/tvBTF' num2str(s, '%02d') '/poly-' num2str(D)];
        mkdir(output_dir);

        path_prefix = ['data-staf/tvBTF' num2str(s, '%02d')];
        configfile = [path_prefix '/tvBTF' num2str(s, '%02d') '-info.txt'];
        [datapath, result_prefix, width, height, timefile] = load_config_file(configfile);
        X = load_sample_timefile(timefile);
        n_time = size(X,2);

        FR = zeros(height, width, 4);
        FG = zeros(height, width, 4);
        FB = zeros(height, width, 4);
        FS = zeros(height, width, 4);
        FI = zeros(height, width, 4);

        fD = 7;
        QR = zeros(fD, 1);
        QG = zeros(fD, 1);
        QB = zeros(fD, 1);
        QS = zeros(fD, 1);
        QI = zeros(fD, 1);

        fprintf('loading files ');
        factors = ['A', 'B', 'C', 'D'];
        for f = 1:4
            img_d = exrread([datapath '\\SIM\\tvBTF' num2str(s, '%02d') '-staf-Kd-' factors(f) '.exr']);        
            img_s = exrread([datapath '\\SIM\\tvBTF' num2str(s, '%02d') '-staf-Ks-' factors(f) '.exr']);        
            img_i = exrread([datapath '\\SIM\\tvBTF' num2str(s, '%02d') '-staf-Sigma-' factors(f) '.exr']);
            FR(:,:,f) = img_d(:,:,1);
            FG(:,:,f) = img_d(:,:,2);
            FB(:,:,f) = img_d(:,:,3);
            FS(:,:,f) = img_s(:,:,1);
            FI(:,:,f) = img_i(:,:,1);
            fprintf('.');
        end

        fid = fopen([datapath '\\SIM\\tvBTF' num2str(s, '%02d') '-staf-Kd-phi.txt'], 'r');
        tline = fgetl(fid);
        while ischar(tline)
            if ~isempty(strfind(tline, 'pphi_r:'))
                p = strsplit(tline);
                p = p';
                for k = 3:size(p,1) - 1
                    q = str2num(p{k});
                    QR(k-2) = q;
                end
            end
            if ~isempty(strfind(tline, 'pphi_g:'))
                p = strsplit(tline);
                p = p';
                for k = 3:size(p,1) - 1
                    q = str2num(p{k});
                    QG(k-2) = q;
                end
            end
            if ~isempty(strfind(tline, 'pphi_b:'))
                p = strsplit(tline);
                p = p';
                for k = 3:size(p,1) - 1
                    q = str2num(p{k});
                    QB(k-2) = q;
                end
            end
            tline = fgetl(fid);
        end
        fclose(fid);

        fid = fopen([datapath '\\SIM\\tvBTF' num2str(s, '%02d') '-staf-Ks-phi.txt'], 'r');
        tline = fgetl(fid);
        while ischar(tline)
            if ~isempty(strfind(tline, 'pphi_Ks:'))
                p = strsplit(tline);
                p = p';
                for k = 3:size(p,1) - 1
                    q = str2num(p{k});
                    QS(k-2) = q;
                end
            end
            if ~isempty(strfind(tline, 'pphi_S:'))
                p = strsplit(tline);
                p = p';
                for k = 3:size(p,1) - 1
                    q = str2num(p{k});
                    QI(k-2) = q;
                end
            end
            tline = fgetl(fid);
        end
        fclose(fid);
        fprintf('done.\n');
        
        Pr = zeros(height, width, D + 1);
        Pg = zeros(height, width, D + 1);
        Pb = zeros(height, width, D + 1);
        Ps = zeros(height, width, D + 1);
        Pi = zeros(height, width, D + 1);
        
        F = cell(5,1);
        F{1} = FR;
        F{2} = FG;
        F{3} = FB;
        F{4} = FS;
        F{5} = FI;

        Q = cell(5,1);
        Q{1} = QR;
        Q{2} = QG;
        Q{3} = QB;
        Q{4} = QS;
        Q{5} = QI;
        
        fprintf('fitting \n');
        n = 50;
        x = linspace(0,1,n); 
        
        for i = 1 : height
            for j = 1 : width
                fy = zeros(5,n);
                for k = 1:5
                    Fj = F{k};
                    q = Q{k};
                    %q = fliplr(q);
                    for l = 1:n
                        t = (x(l) - Fj(i,j,3)) / Fj(i,j,2);
                        fy(k,l) = Fj(i,j,1) * polyval(q,t) + Fj(i,j,4);
                    end
                end

                for l = 1:n
                    if fy(5,l) <= 0
                        fy(5,l) = Inf;
                        fy(4,l) = 0;
                    else
                        fy(5,l) = 1 / sqrt(fy(5,l));
                        fy(4,l) = pi * (fy(4,l) .* (fy(5,l) .^ 2) .* (1 - exp(-1/(fy(5,l).^2))));
                    end
                    fy(5,l) = max(min(1, fy(5,l)), 0);
                    fy(4,l) = max(min(1, fy(4,l)), 0);
                end
                X = x;
                Yr = fy(1,:);
                Yg = fy(2,:);
                Yb = fy(3,:);
                Ys = fy(4,:);
                Yi = fy(5,:);
                Pr(i,j,:) = polyfit(X, Yr, D);
                Pg(i,j,:) = polyfit(X, Yg, D);
                Pb(i,j,:) = polyfit(X, Yb, D);
                Ps(i,j,:) = polyfit(X, Ys, D);
                Pi(i,j,:) = polyfit(X, Yi, D);
            end
            fprintf('%d / %d\n', i, height);
        end
                
        img_d = single(zeros(height, width, 3));
        img_s = single(zeros(height, width));
        img_i = single(zeros(height, width));

        fprintf('saving files ');
        for d = 1 : D + 1
            for i = 1 : height
                for j = 1 : width
                    img_d(i,j,1) = Pr(i,j,d);
                    img_d(i,j,2) = Pg(i,j,d);
                    img_d(i,j,3) = Pb(i,j,d);
                    img_s(i,j) = Ps(i,j,d);
                    img_i(i,j) = Pi(i,j,d);
                end
            end
            exrwrite(img_d, [output_dir '/Diffuse-' num2str(D+1-d) '.exr']);
            exrwrite(img_s, [output_dir '/Specular-' num2str(D+1-d) '.exr']);
            exrwrite(img_i, [output_dir '/Roughness-' num2str(D+1-d) '.exr']);
            fprintf('.');
        end
        fprintf('done.\n');
        
    end
end
