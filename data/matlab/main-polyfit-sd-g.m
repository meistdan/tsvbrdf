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
    %for s = {'AgedChrome', 'DuctTape', 'FractureAsphalt', 'GoldFlake', 'Grass', 'OldWood', 'SnowyGround', 'WoodLog' }
    for s = {'Metal', 'StoneDirt'}
    
        sample = s{1}
        data_path = ['d:/projects/tsvbrdf/data/original/' sample];
        input_dir = [data_path '/frames'];
        output_dir = [data_path '/poly-' num2str(D)];
        mkdir(output_dir);
        
        NR = zeros(height, width, n_time);
        NG = zeros(height, width, n_time);
        NB = zeros(height, width, n_time);
        H = zeros(height, width, n_time);

        fprintf('loading files ');
        for time = 0:n_time-1
            img_n = exrread([input_dir '/Normal-' num2str(time) '.exr']);        
            %img_h = exrread([input_dir '/Height-' num2str(time) '.exr']);        
            NR(:,:,time+1) = img_n(:,:,1);
            NG(:,:,time+1) = img_n(:,:,2);
            NB(:,:,time+1) = img_n(:,:,3);
            %H(:,:,time+1) = img_h(:,:,1);
            fprintf('.');
        end
        fprintf('done.\n');

        Pnr = zeros(height, width, D + 1);
        Png = zeros(height, width, D + 1);
        Pnb = zeros(height, width, D + 1);
        %Ph = zeros(height, width, D + 1);
        
        RMSE_NR = zeros(height, width, n_time);
        RMSE_NG = zeros(height, width, n_time);
        RMSE_NB = zeros(height, width, n_time);
        %RMSE_H = zeros(height, width, n_time);

        fprintf('fitting \n');
        for i = 1 : height
            for j = 1 : width
                a = NR(i,j,:); a = a(:); a = a'; Ynr = a;
                a = NG(i,j,:); a = a(:); a = a'; Yng = a;
                a = NB(i,j,:); a = a(:); a = a'; Ynb = a;
                %a = H(i,j,:); a = a(:); a = a'; Yh = a;
                Pnr(i,j,:) = polyfit(X, Ynr, D);
                Png(i,j,:) = polyfit(X, Yng, D);
                Pnb(i,j,:) = polyfit(X, Ynb, D);
                %Ph(i,j,:) = polyfit(X, Yh, D);
                p = permute(Pnr(i,j,:),[3 2 1]); RMSE_NR(i,j,:) = (polyval(p, X) - Ynr).^ 2;
                p = permute(Png(i,j,:),[3 2 1]); RMSE_NG(i,j,:) = (polyval(p, X) - Yng).^ 2;
                p = permute(Pnb(i,j,:),[3 2 1]); RMSE_NB(i,j,:) = (polyval(p, X) - Ynb).^ 2;
                %p = permute(Ph(i,j,:),[3 2 1]); RMSE_H(i,j,:) = (polyval(p, X) - Yh).^ 2;
            end
            fprintf('%d / %d\n', i, height);
        end
        
        RMSE_NR = sqrt(mean(RMSE_NR(:)));
        RMSE_NG = sqrt(mean(RMSE_NG(:)));
        RMSE_NB = sqrt(mean(RMSE_NB(:)));
        %RMSE_H = sqrt(mean(RMSE_H(:)));

        img_n = single(zeros(height, width, 3));
        %img_h = single(zeros(height, width));

        fprintf('saving files ');
        for d = 1 : D + 1
            for i = 1 : height
                for j = 1 : width
                    img_n(i,j,1) = Pnr(i,j,d);
                    img_n(i,j,2) = Png(i,j,d);
                    img_n(i,j,3) = Pnb(i,j,d);
                    %img_h(i,j) = Ph(i,j,d);
                end
            end
            exrwrite(img_n, [output_dir '/Normal-' num2str(D+1-d) '.exr']);
            %exrwrite(img_h, [output_dir '/Height-' num2str(D+1-d) '.exr']);
            fprintf('.');
        end
        fprintf('done.\n');

        % log RMSE
        %fprintf(fileID, '%s: %f %f %f %f\n', sample, RMSE_NR, RMSE_NG, RMSE_NB, RMSE_H);
    end
    
end
