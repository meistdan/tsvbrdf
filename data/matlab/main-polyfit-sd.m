clc;
close all;
clear all;

cd 'D:/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

n_time = 51;
width = 512;
height = 512;
X = (0:1/(n_time-1):1); 

fileID = fopen('rmse-sd.txt','w');

for D = 3:5
    fprintf(fileID, 'd = %d\n', D);
    %for s = {'AgedChrome', 'DuctTape', 'FractureAsphalt', 'GoldFlake', 'Grass', 'Metal', 'OldWood', 'SnowyGround', 'StoneDirt', 'WoodLog'}
    for s = {'SnowyGround'}
    
        sample = s{1}
        %data_path = ['./data-sd/' sample];
        data_path = ['d:/projects/tsvbrdf/data/original/' sample];
        input_dir = [data_path '/frames'];
        
        output_dir = [data_path '/poly-' num2str(D)];
        mkdir(output_dir);
        
        DR = zeros(height, width, n_time);
        DG = zeros(height, width, n_time);
        DB = zeros(height, width, n_time);
        SR = zeros(height, width, n_time);
        SG = zeros(height, width, n_time);
        SB = zeros(height, width, n_time);
        I = zeros(height, width, n_time);

        fprintf('loading files ');
        for time = 0:n_time-1
            img_d = exrread([input_dir '/Diffuse-' num2str(time) '.exr']);        
            img_s = exrread([input_dir '/Specular-' num2str(time) '.exr']);        
            img_i = exrread([input_dir '/Roughness-' num2str(time) '.exr']);        
            DR(:,:,time+1) = img_d(:,:,1);
            DG(:,:,time+1) = img_d(:,:,2);
            DB(:,:,time+1) = img_d(:,:,3);
            SR(:,:,time+1) = img_s(:,:,1);
            SG(:,:,time+1) = img_s(:,:,2);
            SB(:,:,time+1) = img_s(:,:,3);
            I(:,:,time+1) = img_i(:,:,1);
            fprintf('.');
        end
        fprintf('done.\n');

        Pdr = zeros(height, width, D + 1);
        Pdg = zeros(height, width, D + 1);
        Pdb = zeros(height, width, D + 1);
        Psr = zeros(height, width, D + 1);
        Psg = zeros(height, width, D + 1);
        Psb = zeros(height, width, D + 1);
        Pi = zeros(height, width, D + 1);
        
        RMSE_DR = zeros(height, width, n_time);
        RMSE_DG = zeros(height, width, n_time);
        RMSE_DB = zeros(height, width, n_time);
        RMSE_SR = zeros(height, width, n_time);
        RMSE_SG = zeros(height, width, n_time);
        RMSE_SB = zeros(height, width, n_time);
        RMSE_I = zeros(height, width, n_time);

        fprintf('fitting \n');
        for i = 1 : height
            for j = 1 : width
                a = DR(i,j,:); a = a(:); a = a'; Ydr = a;
                a = DG(i,j,:); a = a(:); a = a'; Ydg = a;
                a = DB(i,j,:); a = a(:); a = a'; Ydb = a;
                a = SR(i,j,:); a = a(:); a = a'; Ysr = a;
                a = SG(i,j,:); a = a(:); a = a'; Ysg = a;
                a = SB(i,j,:); a = a(:); a = a'; Ysb = a;
                a = I(i,j,:); a = a(:); a = a'; Yi = a;
                Pdr(i,j,:) = polyfit(X, Ydr, D);
                Pdg(i,j,:) = polyfit(X, Ydg, D);
                Pdb(i,j,:) = polyfit(X, Ydb, D);
                Psr(i,j,:) = polyfit(X, Ysr, D);
                Psg(i,j,:) = polyfit(X, Ysg, D);
                Psb(i,j,:) = polyfit(X, Ysb, D);
                Pi(i,j,:) = polyfit(X, Yi, D);
                p = permute(Pdr(i,j,:),[3 2 1]); RMSE_DR(i,j,:) = (polyval(p, X) - Ydr).^ 2;
                if abs(polyval(p,0) - Ydr(i)) > 0.01
                    x = linspace(0,1);
                    y = polyval(p,x);
                    figure
                    plot(X,Ydr,'o')
                    hold on
                    plot(x,y)
                    plot(x,zeros(size(x)))
                    hold off
                    pause
                end
                p = permute(Pdg(i,j,:),[3 2 1]); RMSE_DG(i,j,:) = (polyval(p, X) - Ydg).^ 2;
                p = permute(Pdb(i,j,:),[3 2 1]); RMSE_DB(i,j,:) = (polyval(p, X) - Ydb).^ 2;
                p = permute(Psr(i,j,:),[3 2 1]); RMSE_SR(i,j,:) = (polyval(p, X) - Ysr).^ 2;
                p = permute(Psg(i,j,:),[3 2 1]); RMSE_SG(i,j,:) = (polyval(p, X) - Ysg).^ 2;
                p = permute(Psb(i,j,:),[3 2 1]); RMSE_SB(i,j,:) = (polyval(p, X) - Ysb).^ 2;
                p = permute(Pi(i,j,:),[3 2 1]); RMSE_I(i,j,:) = (polyval(p, X) - Yi).^ 2;
                p = permute(Pi(i,j,:),[3 2 1]);
            end
            fprintf('%d / %d\n', i, height);
        end
        
        RMSE_DR = sqrt(mean(RMSE_DR(:)));
        RMSE_DG = sqrt(mean(RMSE_DG(:)));
        RMSE_DB = sqrt(mean(RMSE_DB(:)));
        RMSE_SR = sqrt(mean(RMSE_SR(:)));
        RMSE_SG = sqrt(mean(RMSE_SG(:)));
        RMSE_SB = sqrt(mean(RMSE_SB(:)));
        RMSE_I = sqrt(mean(RMSE_I(:)));

        img_d = single(zeros(height, width, 3));
        img_s = single(zeros(height, width, 3));
        img_i = single(zeros(height, width));

        fprintf('saving files ');
        for d = 1 : D + 1
            for i = 1 : height
                for j = 1 : width
                    img_d(i,j,1) = Pdr(i,j,d);
                    img_d(i,j,2) = Pdg(i,j,d);
                    img_d(i,j,3) = Pdb(i,j,d);
                    img_s(i,j,1) = Psr(i,j,d);
                    img_s(i,j,2) = Psg(i,j,d);
                    img_s(i,j,3) = Psb(i,j,d);
                    img_i(i,j) = Pi(i,j,d);
                end
            end
%             exrwrite(img_d, [output_dir '/Diffuse-' num2str(D+1-d) '.exr']);
%             exrwrite(img_s, [output_dir '/Specular-' num2str(D+1-d) '.exr']);
%             exrwrite(img_i, [output_dir '/Roughness-' num2str(D+1-d) '.exr']);
            fprintf('.');
        end
        fprintf('done.\n');

        % log RMSE
        fprintf(fileID, '%s: %f %f %f %f %f %f %f\n', sample, RMSE_DR, RMSE_DG, RMSE_DB, RMSE_SR, RMSE_SG, RMSE_SB, RMSE_I);
    end
    
end
