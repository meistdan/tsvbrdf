clc;
close all;
clear all;

cd 'C:/Users/meistdan/Documents/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

n_time = 51;

for D = 5:5
    %for s  = [09 20 22 23 24 25 27 29 30 31 32 35 37 39 40 41 42 43 44 45]
    for s = [25]
        
        fprintf('sample %d\n', s);
        input_dir = ['data-staf/tvBTF' num2str(s, '%02d') '/poly-' num2str(D)];
        output_dir = ['./data-staf/tvBTF' num2str(s, '%02d') '/frames-reconstruct-' num2str(D)];
        mkdir(output_dir);
        
        path_prefix = ['data-staf/tvBTF' num2str(s, '%02d')];
        configfile = [path_prefix '/tvBTF' num2str(s, '%02d') '-info.txt'];
        [datapath, result_prefix, width, height, timefile] = load_config_file(configfile);
        
        Pr = zeros(height, width, D + 1);
        Pg = zeros(height, width, D + 1);
        Pb = zeros(height, width, D + 1);
        Ps = zeros(height, width, D + 1);
        Pi = zeros(height, width, D + 1);
        
        fprintf('loading files ');
        for d = 1 : D + 1
            diffuse = exrread([input_dir '/Diffuse-' num2str(D+1-d) '.exr']);
            specular = exrread([input_dir '/Specular-' num2str(D+1-d) '.exr']);
            roughness = exrread([input_dir '/Roughness-' num2str(D+1-d) '.exr']);
            Pr(:,:,d) = diffuse(:,:,1);
            Pg(:,:,d) = diffuse(:,:,2);
            Pb(:,:,d) = diffuse(:,:,3);
            Ps(:,:,d) = specular(:,:,1);
            Pi(:,:,d) = roughness(:,:,1);
            fprintf('.');
        end
        
        img_d = single(zeros(height, width, 3));
        img_s = single(zeros(height, width));
        img_i = single(zeros(height, width));
        X = linspace(0,1,51);
        for time = 0:n_time-1
            for i = 1 : height
                for j = 1 : width
                    x = X(time+1);
                    pr = permute(Pr(i,j,:), [3 2 1]);
                    pg = permute(Pg(i,j,:), [3 2 1]);
                    pb = permute(Pb(i,j,:), [3 2 1]);
                    ps = permute(Ps(i,j,:), [3 2 1]);
                    pi = permute(Pi(i,j,:), [3 2 1]);
                    img_d(i,j,1) = polyval(pr,x);
                    img_d(i,j,2) = polyval(pg,x);
                    img_d(i,j,3) = polyval(pb,x);
                    img_s(i,j) = polyval(ps,x);
                    img_i(i,j) = polyval(pi,x);
                end
                fprintf('.');
            end
            exrwrite(img_d, [output_dir '/Diffuse-' num2str(time) '.exr']);
            exrwrite(img_s, [output_dir '/Specular-' num2str(time) '.exr']);
            exrwrite(img_i, [output_dir '/Roughness-' num2str(time) '.exr']);
        end
        fprintf('done.\n');
        
    end
    
end
