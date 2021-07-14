clc;
close all;
clear all;

cd 'c:/Users/meist/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

border = 20;

samples = [41 42 44 45]';
sample_count = size(samples, 1)

sample_labels = cell(sample_count, 1);
% sample_labels{1} = 'Wood Burning'; %09
% sample_labels{2} = 'Orange Cloth Drying'; %22
% sample_labels{3} = 'Light Wood Drying'; %23
% sample_labels{4} = 'White Felt Drying'; %24
% sample_labels{5} = 'Quilted Paper Drying'; %25
% sample_labels{6} = 'Wet Brick Drying'; %27
% sample_labels{7} = 'Wood Drying'; %29
% sample_labels{8} = 'Green Cloth Drying'; %30
% sample_labels{9} = 'Banana Decaying'; %31
% sample_labels{10} = 'Steel Rusting'; %32
% sample_labels{11} = 'Leaf Drying'; %33
% sample_labels{12} = 'Apple Slice Decaying'; %35
% sample_labels{13} = 'Granite Drying'; %37
% sample_labels{14} = 'Potato Decaying'; %39
% sample_labels{15} = 'Charred Wood Burning'; %40
% sample_labels{16} = 'Waffle Toasting'; %41
% sample_labels{17} = 'Bread Toasting'; %42
% sample_labels{18} = 'Copper Patina',; %44
% sample_labels{19} = 'Cast Iron Casting'; %45

sample_labels{1} = 'Waffle Toasting'; %41
sample_labels{2} = 'Bread Toasting'; %42
sample_labels{3} = 'Copper Patina',; %44
sample_labels{4} = 'Cast Iron Casting'; %45

for si = 1 : sample_count
    
    stats = cell(3, 8);
    
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
    fprintf('done.\n');
    
    fprintf('normalizing Ks and sigma ');
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
            end
        end
    end

    for D = 3:5

        output_dir = ['data-staf/tvBTF' num2str(s, '%02d') '/poly-' num2str(D)];
        mkdir(output_dir);
        
        Pr = zeros(height, width, D + 1);
        Pg = zeros(height, width, D + 1);
        Pb = zeros(height, width, D + 1);
        Ps = zeros(height, width, D + 1);
        Pi = zeros(height, width, D + 1);
        
        RMSE_R = zeros(height, width, n_time);
        RMSE_G = zeros(height, width, n_time);
        RMSE_B = zeros(height, width, n_time);
        RMSE_S = zeros(height, width, n_time);
        RMSE_I = zeros(height, width, n_time);
        
        fprintf('fitting \n');
        tic;
        for i = 1 : height
            for j = 1 : width
                a = R(i,j,:); a = a(:); a = a'; Yr = a;
                a = G(i,j,:); a = a(:); a = a'; Yg = a;
                a = B(i,j,:); a = a(:); a = a'; Yb = a;
                a = S(i,j,:); a = a(:); a = a'; Ys = a;
                a = I(i,j,:); a = a(:); a = a'; Yi = a;
                Pr(i,j,:) = polyfit(X, Yr, D);
                Pg(i,j,:) = polyfit(X, Yg, D);
                Pb(i,j,:) = polyfit(X, Yb, D);
                Ps(i,j,:) = polyfit(X, Ys, D);
                Pi(i,j,:) = polyfit(X, Yi, D);
                % RMSE
                p = permute(Pr(i,j,:),[3 2 1]); RMSE_R(i,j,:) = (polyval(p, X) - Yr).^ 2;
                p = permute(Pg(i,j,:),[3 2 1]); RMSE_G(i,j,:) = (polyval(p, X) - Yg).^ 2;
                p = permute(Pb(i,j,:),[3 2 1]); RMSE_B(i,j,:) = (polyval(p, X) - Yb).^ 2;
                p = permute(Ps(i,j,:),[3 2 1]); RMSE_S(i,j,:) = (polyval(p, X) - Ys).^ 2;
                p = permute(Pi(i,j,:),[3 2 1]); RMSE_I(i,j,:) = (polyval(p, X) - Yi).^ 2;
            end
            fprintf('%d / %d\n', i, height);
        end
        fit_time = toc;
        
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
        
        % log RMSE and time
        stats{D - 2, 1} = ['tvBTF' num2str(s, '%02d')]
        stats{D - 2, 2} = sample_label;
        stats{D - 2, 3} = RMSE_R;
        stats{D - 2, 4} = RMSE_G;
        stats{D - 2, 5} = RMSE_B;
        stats{D - 2, 6} = RMSE_S;
        stats{D - 2, 7} = RMSE_I;
        stats{D - 2, 8} = fit_time;
    end
    
    save(['stats-staf-poly-crop-' sample_label '.mat'], 'stats');
end
