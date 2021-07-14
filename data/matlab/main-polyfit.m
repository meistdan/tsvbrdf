clc;
close all;
clear all;

cd 'C:/Users/meistdan/Documents/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

for D = 3:5
    for s  = [09 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35 37 38 39 40 41 42 43 44 45]
    %for s = [09]

        fprintf('sample %d\n', s);
        output_dir = ['d:/projects/tsvbrdf/data/original/tvBTF' num2str(s, '%02d') '/poly-' num2str(D)];
        mkdir(output_dir);

        path_prefix = ['d:/projects/tsvbrdf/data/original/tvBTF' num2str(s, '%02d')];
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

        Pr = zeros(height, width, D + 1);
        Pg = zeros(height, width, D + 1);
        Pb = zeros(height, width, D + 1);
        Ps = zeros(height, width, D + 1);
        Pi = zeros(height, width, D + 1);

        fprintf('fitting \n');
        for i = 1 : height
            mm = 0;
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
                p = Pi(i,j,:);
                p = permute(p,[3 2 1]);
                if sum(isinf(p))
                    disp(p);
                end
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
            exrwrite(img_d, [output_dir '/Kd-' num2str(D+1-d) '.exr']);
            exrwrite(img_s, [output_dir '/Ks-' num2str(D+1-d) '.exr']);
            exrwrite(img_i, [output_dir '/Sigma-' num2str(D+1-d) '.exr']);
            fprintf('.');
        end
        fprintf('done.\n');

    end
    
end
