clc;
close all;
clear all;

cd 'c:/Users/meist/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

samples = [09 22 23 24 25 27 29 30 31 32 33 35 37 39 40 41 42 44 45]';
sample_count = size(samples, 1)

sample_labels = cell(sample_count, 1);
sample_labels{1} = 'Wood Burning'; %09
sample_labels{2} = 'Orange Cloth Drying'; %22
sample_labels{3} = 'Light Wood Drying'; %23
sample_labels{4} = 'White Felt Drying'; %24
sample_labels{5} = 'Quilted Paper Drying'; %25
sample_labels{6} = 'Wet Brick Drying'; %27
sample_labels{7} = 'Wood Drying'; %29
sample_labels{8} = 'Green Cloth Drying'; %30
sample_labels{9} = 'Banana Decaying'; %31
sample_labels{10} = 'Steel Rusting'; %32
sample_labels{11} = 'Leaf Drying'; %33
sample_labels{12} = 'Apple Slice Decaying'; %35
sample_labels{13} = 'Granite Drying'; %37
sample_labels{14} = 'Potato Decaying'; %39
sample_labels{15} = 'Charred Wood Burning'; %40
sample_labels{16} = 'Waffle Toasting'; %41
sample_labels{17} = 'Bread Toasting'; %42
sample_labels{18} = 'Copper Patina',; %44
sample_labels{19} = 'Cast Iron Casting'; %45

border = 20;

for si = 1 : sample_count

    stats = cell(1, 7);
    
    s = samples(si);
    sample_label = sample_labels{si};
    fprintf('sample %d\n', s);
    
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
    
    RMSE_R = zeros(height, width, n_time);
    RMSE_G = zeros(height, width, n_time);
    RMSE_B = zeros(height, width, n_time);
    RMSE_S = zeros(height, width, n_time);
    RMSE_I = zeros(height, width, n_time);
    
    for i = 1 : height
        for j = 1 : width
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
                
                t0 = (X(t) - FR(i,j,3)) / FR(i,j,2); fr = FR(i,j,1) * polyval(QR,t0) + FR(i,j,4); 
                t0 = (X(t) - FG(i,j,3)) / FG(i,j,2); fg = FG(i,j,1) * polyval(QG,t0) + FG(i,j,4);
                t0 = (X(t) - FB(i,j,3)) / FB(i,j,2); fb = FB(i,j,1) * polyval(QB,t0) + FB(i,j,4);
                t0 = (X(t) - FS(i,j,3)) / FS(i,j,2); fs = FS(i,j,1) * polyval(QS,t0) + FS(i,j,4);
                t0 = (X(t) - FI(i,j,3)) / FI(i,j,2); fi = FI(i,j,1) * polyval(QI,t0) + FI(i,j,4);
                
                if fi <= 0
                    fi = Inf;
                    fs = 0;
                else
                    fi = 1 / sqrt(fi);
                    fs = pi * (fs * (fi^2) * (1 - exp(-1/(fi^2))));
                end
                fi = max(min(1, fi), 0);
                fs = max(min(1, fs), 0);
                
                RMSE_R(i,j,t) = (fr - R(i,j,t)) ^ 2;
                RMSE_G(i,j,t) = (fg - G(i,j,t)) ^ 2;
                RMSE_B(i,j,t) = (fb - B(i,j,t)) ^ 2;
                RMSE_S(i,j,t) = (fs - S(i,j,t)) ^ 2;
                RMSE_I(i,j,t) = (fi - I(i,j,t)) ^ 2;

            end
        end
        fprintf('%d / %d\n', i, height);
    end
    
    RMSE_R = RMSE_R(1 + border : height - border, 1 + border : width - border);
    RMSE_G = RMSE_G(1 + border : height - border, 1 + border : width - border);
    RMSE_B = RMSE_B(1 + border : height - border, 1 + border : width - border);
    RMSE_S = RMSE_S(1 + border : height - border, 1 + border : width - border);
    RMSE_I = RMSE_I(1 + border : height - border, 1 + border : width - border);
    
    RMSE_R = sqrt(mean(RMSE_R(:)));
    RMSE_G = sqrt(mean(RMSE_G(:)));
    RMSE_B = sqrt(mean(RMSE_B(:)));
    RMSE_S = sqrt(mean(RMSE_S(:)));
    RMSE_I = sqrt(mean(RMSE_I(:)));

    stats{1, 1} = ['tvBTF' num2str(s, '%02d')];
    stats{1, 2} = sample_label;
    stats{1, 3} = RMSE_R;
    stats{1, 4} = RMSE_G;
    stats{1, 5} = RMSE_B;
    stats{1, 6} = RMSE_S;
    stats{1, 7} = RMSE_I;

    save(['stats-staf-gu-' sample_label '.mat'], 'stats');
end


fprintf('done.\n');
