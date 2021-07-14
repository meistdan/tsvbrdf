clc;
close all;
clear all;

cd 'C:/Users/meistdan/Documents/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

width = 512;
height = 512;
n_time = 51;

for D = 5:5
    for s = {'AgedChrome' }
        
        sample = s{1}
        input_dir = ['data-sd/' sample '/poly-' num2str(D)];
        output_dir = ['./data-sd/' sample '/frames-reconstruct-' num2str(D)];
        mkdir(output_dir);
        
        path_prefix = ['data-sd/' sample];
                
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
