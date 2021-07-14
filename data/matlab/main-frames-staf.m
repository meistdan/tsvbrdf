clc;
close all;
clear all;

cd 'C:/Users/meistdan/Documents/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

frames = 51;

for s = [09 20 22 23 24 25 27 29 30 31 32 35 37 39 40 41 42 43 44 45]

    fprintf('sample %d\n', s);
    output_dir = ['./data-staf/tvBTF' num2str(s, '%02d') '/frames'];
    mkdir(output_dir);

    path_prefix = ['./data-staf/tvBTF' num2str(s, '%02d')];
    configfile = [path_prefix '/tvBTF' num2str(s, '%02d') '-info.txt'];
    [datapath, result_prefix, width, height, timefile] = load_config_file(configfile);
    X = load_sample_timefile(timefile)';
    n_time = size(X,1);

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

    Pr = zeros(height, width, frames);
    Pg = zeros(height, width, frames);
    Pb = zeros(height, width, frames);
    Ps = zeros(height, width, frames);
    Pi = zeros(height, width, frames);

    fprintf('resampling \n');
    Xi = (0:1/(frames-1):1)'; 
    for i = 1 : height
        for j = 1 : width
            a = R(i,j,:); a = a(:); Yr = a;
            a = G(i,j,:); a = a(:); Yg = a;
            a = B(i,j,:); a = a(:); Yb = a;
            a = S(i,j,:); a = a(:); Ys = a;
            a = I(i,j,:); a = a(:); Yi = a;
            Pr(i,j,:) = interp1q(X, Yr, Xi);
            Pg(i,j,:) = interp1q(X, Yg, Xi);
            Pb(i,j,:) = interp1q(X, Yb, Xi);
            Ps(i,j,:) = interp1q(X, Ys, Xi);
            Pi(i,j,:) = interp1q(X, Yi, Xi);
        end
        %fprintf('%d / %d\n', i, height);
    end

    img_d = single(zeros(height, width, 3));
    img_s = single(zeros(height, width, 3));
    img_i = single(zeros(height, width, 3));

    fprintf('saving files ');
    for d = 1 : frames
        for i = 1 : height
            for j = 1 : width
                if Pi(i,j,d) <= 0
                    Pi(i,j,d) = Inf;
                    Ps(i,j,d) = 0;
                else
                    Pi(i,j,d) = 1 / sqrt(Pi(i,j,d));
                    Ps(i,j,d) = pi * (Ps(i,j,d) .* (Pi(i,j,d) .^ 2) .* (1 - exp(-1/(Pi(i,j,d).^2))));
                end
                Pi(i,j,d) = max(min(1, Pi(i,j,d)), 0);
                Ps(i,j,d) = max(min(1, Ps(i,j,d)), 0);
                img_d(i,j,1) = Pr(i,j,d);
                img_d(i,j,2) = Pg(i,j,d);
                img_d(i,j,3) = Pb(i,j,d);
                img_s(i,j,1) = Ps(i,j,d);
                img_s(i,j,2) = Ps(i,j,d);
                img_s(i,j,3) = Ps(i,j,d);
                img_i(i,j,1) = Pi(i,j,d);
                img_i(i,j,2) = Pi(i,j,d);
                img_i(i,j,3) = Pi(i,j,d);
            end
        end
        exrwrite(img_d, [output_dir '/Diffuse-' num2str(d-1) '.exr']);
        exrwrite(img_s, [output_dir '/Specular-' num2str(d-1) '.exr']);
        exrwrite(img_i, [output_dir '/Roughness-' num2str(d-1) '.exr']);
        fprintf('.');
    end
    fprintf('done.\n');
end