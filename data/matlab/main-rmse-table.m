clc;
close all;
clear all;

cd 'c:/Users/meist/projects/MATLAB/tsvbrdf';

samples = [09 22 23 24 25 27 29 30 31 32 35 37 39 40 41 42 44 45]';
sample_count = size(samples, 1);

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
sample_labels{11} = 'Apple Slice Decaying'; %35
sample_labels{12} = 'Granite Drying'; %37
sample_labels{13} = 'Potato Decaying'; %39
sample_labels{14} = 'Charred Wood Burning'; %40
sample_labels{15} = 'Waffle Toasting'; %41
sample_labels{16} = 'Bread Toasting'; %42
sample_labels{17} = 'Copper Patina'; %44
sample_labels{18} = 'Cast Iron Casting'; %45

sample_res = cell(sample_count, 1);
sample_res{1} = '$512^2 \times 10$'; %09
sample_res{2} = '$512^2 \times 33$'; %22
sample_res{3} = '$300^2 \times 34$'; %23
sample_res{4} = '$300^2 \times 28$'; %24
sample_res{5} = '$300^2 \times 25$'; %25
sample_res{6} = '$260^2 \times 32$'; %27
sample_res{7} = '$420^2 \times 14$'; %29
sample_res{8} = '$420^2 \times 30$'; %30
sample_res{9} = '$220^2 \times 33$'; %31
sample_res{10} = '$300^2 \times 22$'; %32
sample_res{11} = '$156^2 \times 35$'; %35
sample_res{12} = '$340^2 \times 27$'; %37
sample_res{13} = '$200^2 \times 36$'; %39
sample_res{14} = '$480^2 \times 31$'; %40
sample_res{15} = '$220^2 \times 30$'; %41
sample_res{16} = '$260^2 \times 30$'; %42
sample_res{17} = '$460^2 \times 34$'; %44
sample_res{18} = '$460^2 \times 30$'; %45

fileID = fopen('table.tex','w');

comp_labels = cell(5, 1);
comp_labels{1} = '$K_d^\\star$ (R)';
comp_labels{2} = '$K_d^\\star$ (G)';
comp_labels{3} = '$K_d^\\star$ (B)';
comp_labels{4} = '$K_s^\\star$';
comp_labels{5} = '$\\sigma^\\star$';

fileID_avg = fopen('rmse.txt','w');

gu_avg = 0;
d3_avg = 0;
d4_avg = 0;
d5_avg = 0;

for si = 1 : sample_count
    
    s = samples(si);
    sample_label = sample_labels{si};
    sample_re = sample_res{si};
    sample_code = ['tvBTF' num2str(s, '%02d')];

    gu = load(['stats-staf-gu-' sample_label '.mat']);
    poly = load(['stats-staf-poly-crop-' sample_label '.mat']);
    
    fprintf(fileID,'\\rotatebox{90}{%s} &\n', sample_label);    
    fprintf(fileID,'\\makecell{\\includegraphics[width=0.085\\textwidth,trim=0 0 0 0,clip]{images/enlargement/%s/poly-5/original/0}} &\n', sample_code);    
    
    fprintf(fileID,'\\begin{tabular}{c}\n');    
    fprintf(fileID,'%s\\\\ STAF\\\\ $d = 3$\\\\ $d = 4$\\\\ $d = 5$\\\\\n', sample_re);    
    fprintf(fileID,'\\end{tabular} &\n');    
    
    for j = 3:7
        p0 = gu.stats{1, j};
        p1 = poly.stats{1, j};
        p2 = poly.stats{2, j};
        p3 = poly.stats{3, j};
        f0 = '%.3f';
        f1 = '%.3f';
        f2 = '%.3f';
        f3 = '%.3f';
        if p0 < 0.001
           f0 = '%.0e'; 
        end
        if p1 < 0.001
           f1 = '%.0e'; 
        end
        if p2 < 0.001
           f2 = '%.0e'; 
        end
        if p3 < 0.001
           f3 = '%.0e'; 
        end
        fprintf(fileID,'\\begin{tabular}{c}\n');
        fprintf(fileID,[comp_labels{j - 2} '\\\\' f0 '\\\\ ' f1 '\\\\ ' f2 '\\\\ ' f3 '\\\\\n'], gu.stats{1, j}, poly.stats{1, j}, poly.stats{2, j}, poly.stats{3, j});    
        fprintf(fileID,'\\end{tabular} &\n');
    end
    
    
    fprintf(fileID,'\\begin{tabular}{c}\n');    
    fprintf(fileID,'time [s]\\\\ -\\\\ %.0f\\\\ %.0f\\\\ %.0f\\\\\n', poly.stats{1, 8}, poly.stats{2, 8}, poly.stats{3, 8});    
    fprintf(fileID,'\\end{tabular}');

    if mod(si, 2) == 0
        fprintf(fileID,'\\\\\n');
    else
        fprintf(fileID,'&\n');
    end
    
    fprintf(fileID_avg, "%s\n", sample_label);
    fprintf(fileID_avg, "STAF avg. RMSE\t %f\n", (gu.stats{1, 3} + gu.stats{1, 4}+ gu.stats{1, 5} + gu.stats{1, 6} + gu.stats{1, 7}) / 5);
    fprintf(fileID_avg, "d=3 avg. RMSE\t %f\n", (poly.stats{1, 3} + poly.stats{1, 4}+ poly.stats{1, 5} + poly.stats{1, 6} + poly.stats{1, 7}) / 5);
    fprintf(fileID_avg, "d=4 avg. RMSE\t %f\n", (poly.stats{2, 3} + poly.stats{2, 4}+ poly.stats{2, 5} + poly.stats{2, 6} + poly.stats{2, 7}) / 5);
    fprintf(fileID_avg, "d=5 avg. RMSE\t %f\n", (poly.stats{3, 3} + poly.stats{3, 4}+ poly.stats{3, 5} + poly.stats{3, 6} + poly.stats{3, 7}) / 5);
    fprintf(fileID_avg, "\n");
    
    gu_avg = gu_avg + gu.stats{1, 3} + gu.stats{1, 4}+ gu.stats{1, 5} + gu.stats{1, 6} + gu.stats{1, 7};
    d3_avg = d3_avg + poly.stats{1, 3} + poly.stats{1, 4}+ poly.stats{1, 5} + poly.stats{1, 6} + poly.stats{1, 7};
    d4_avg = d4_avg + poly.stats{2, 3} + poly.stats{2, 4}+ poly.stats{2, 5} + poly.stats{2, 6} + poly.stats{2, 7};
    d5_avg = d5_avg + poly.stats{3, 3} + poly.stats{3, 4}+ poly.stats{3, 5} + poly.stats{3, 6} + poly.stats{3, 7};
    
end

fprintf(fileID_avg, "Overall\n");
fprintf(fileID_avg, "STAF avg. RMSE\t %f\n", gu_avg / (sample_count * 5));
fprintf(fileID_avg, "d=3 avg. RMSE\t %f\n", d3_avg / (sample_count * 5));
fprintf(fileID_avg, "d=4 avg. RMSE\t %f\n", d4_avg / (sample_count * 5));
fprintf(fileID_avg, "d=5 avg. RMSE\t %f\n", d5_avg / (sample_count * 5));
fprintf(fileID_avg, "\n");

fclose(fileID);
fclose(fileID_avg);