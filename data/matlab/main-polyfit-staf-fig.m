clc;
close all;
clear all;

cd 'c:/Users/meist/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

seed = 3;
rng(seed, 'twister');

border = 20;

S0 = [
   103   96    39   159   280;
    94   64    86    68   172;
];

S1 = [
   295    181   256    86   115;
   322    182   264   144   329;
];

SP = cell(2,1);
SP{1} = S0;
SP{2} = S1;

%samples = [45];
samples = [29, 31, 37, 41, 45]

sidx = 1;
for s = samples

    fprintf('sample %d\n', s);
    output_dir = 'c:\Users\meist\Documents\TRL\papers\TSVBRDF\images\fitting\';
    mkdir(output_dir);

    path_prefix = ['data-staf/tvBTF' num2str(s, '%02d')];
    configfile = [path_prefix '/tvBTF' num2str(s, '%02d') '-info.txt'];
    [datapath, result_prefix, width, height, timefile] = load_config_file(configfile);
    X = load_sample_timefile(timefile);
    n_time = size(X,2);

    R = zeros(height, width, n_time);
    G = zeros(height, width, n_time);
    B = zeros(height, width, n_time);
    S = zeros(height, width, n_time);
    I = zeros(height, width, n_time);
    
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
    for time = 0:n_time-1
        img_d = exrread([datapath '\\TSparamNew\\' result_prefix '-' num2str(time, '%02d') '-xx-TSparam-Kd.exr']);        
        img_s = exrread([datapath '\\TSparamNew\\' result_prefix '-' num2str(time, '%02d') '-xx-TSparam-Ks.exr']);        
        img_i = exrread([datapath '\\TSparamNew\\' result_prefix '-' num2str(time, '%02d') '-xx-TSparam-Sigma.exr']);
        R(:,:,time+1) = img_d(:,:,1);
        G(:,:,time+1) = img_d(:,:,2);
        B(:,:,time+1) = img_d(:,:,3);
        S(:,:,time+1) = img_s(:,:,1);
        I(:,:,time+1) = img_i(:,:,1);
        fprintf('.');
    end
    
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
    
    for i = 1:height
        for j = 1:width
            for t = 1:n_time
                if I(i,j,t) <= 0
                    I(i,j,t) = Inf;
                    S(i,j,t) = 0;
                else
                    I(i,j,t) = 1 / sqrt(I(i,j,t));
                    S(i,j,t) = pi * (S(i,j,t) .* (I(i,j,t) .^ 2) .* (1 - exp(-1/(I(i,j,t).^2))));
                end
                I(i,j,t) = max(min(1, I(i,j,t)), 0);
                S(i,j,t) = max(min(1, S(i,j,t)), 0);
            end
        end
    end

    P = cell(5,1);
    P{1} = R;
    P{2} = G;
    P{3} = B;
    P{4} = S;
    P{5} = I;

    for pt = 1:2
        %i = randi([1 + border, height - border]);
        %j = randi([1 + border ,width - border]);
        
        Si = SP{pt};
        i = Si(1,sidx);
        j = Si(2,sidx);

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
        n = 200;
        x = linspace(0,1,n); 

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

        colors = zeros(5,3);
        colors(1,:) = [0, 0.4470, 0.7410];
        colors(2,:) = [0.8500, 0.3250, 0.0980];
        colors(3,:) = [0.9290, 0.6940, 0.1250];
        colors(4,:) = [0.4940, 0.1840, 0.5560];
        colors(5,:) = [0.4660, 0.6740, 0.1880];

        for k = 1:5
            Pj = P{k};
            a = Pj(i,j,:); a = a(:); a = a'; Y = a;
            figure('visible','off','Position', [0 0 200 100]);
            grid on;
            hold on;
            plot(X, Y,'x', 'LineWidth', 2, 'Color', colors(1,:));
            for D = 3:5
                Pf = polyfit(X, Y, D);
                p = permute(Pf,[3 2 1]);
                y = polyval(p,x);
                plot(x,y, 'LineWidth', 2, 'Color', colors(D-1,:));
            end
            plot(x,fy(k,:), 'LineWidth', 2, 'Color', colors(5,:));
            %legend('raw data','d = 3', 'd = 4', 'd = 5');
            %if k <= 3
                %ylim([0,1]);
            %end
            %xlabel('time');
            hold off;
            fig2pdf(gcf, [output_dir 'tvBTF' num2str(s, '%02d') '-' num2str(pt) '-' num2str(k) '-curves.pdf']);
        end
    end
    sidx = sidx + 1;
end

rng(seed, 'twister');
sidx = 1;
for s = samples
    path_prefix = ['data-staf/tvBTF' num2str(s, '%02d')];
    configfile = [path_prefix '/tvBTF' num2str(s, '%02d') '-info.txt'];
    [datapath, result_prefix, width, height, timefile] = load_config_file(configfile);
    
    Pj = imread(['c:\Users\meist\projects\MATLAB\tsvbrdf\data-staf\tvBTF' num2str(s, '%02d') '/poly-5/images/0.jpg']);
    Pj = imresize(Pj,[512 512]);
    figure('visible','off');
    imshow(Pj);
    hold on;
    for pt = 1:2
        %i = randi([1 + border, height - border]);
        %j = randi([1 + border ,width - border]);
        Si = SP{pt};
        i = Si(1,sidx);
        j = Si(2,sidx);
        si = int16(512 / height * i);
        sj = int16(512 / width * j);
        if pt == 1
            plot(si, sj,'x', 'MarkerSize', 50, 'LineWidth', 5, 'Color', 'green');
        else
            plot(si, sj,'x', 'MarkerSize', 50, 'LineWidth', 5, 'Color', 'green');
        end
    end
    Pj = getframe;
    Pj = Pj.cdata;
    imwrite(Pj, [output_dir 'tvBTF' num2str(s, '%02d') '-preview.png']);
    sidx = sidx + 1;
end

fprintf('done.\n');
