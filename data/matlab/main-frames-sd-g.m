clc;
close all;
clear all;

cd 'D:/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

frames = 51;

%for s = {'AgedChrome', 'CanyonRock', 'CottonFabric', 'DryMud', 'DuctTape', 'FractureAsphalt', 'GoldFlake', 'Grass', 'Lemon', 'Metal', 'OldWood', 'SnowyGround', 'StoneDirt', 'SyntheticFoam', 'WhiskeyBarrelOak', 'WoodLog' }
%for s = {'AgedChrome', 'DuctTape', 'FractureAsphalt', 'GoldFlake', 'Grass', 'OldWood', 'SnowyGround', 'WoodLog' }
for s = {'Metal', 'StoneDirt'}
    sample = s{1}
    data_path = ['./data-sd/' sample];
    input_dir = [data_path '/export'];
    output_dir = ['d:/projects/tsvbrdf/data/original/' sample '/frames'];
    %rmdir(output_dir, 's');
    mkdir(output_dir);
    for i = 0:50
        % input
        %height = imread([input_dir '/Height-' num2str(i) '.png']);
        %height = single(height) / single(intmax(class(height)));
        normal = imread([input_dir '/Normal-' num2str(i) '.png']);
        normal = single(normal) / single(intmax(class(normal)));
        % output
        %exrwrite(height, [output_dir '/Height-' num2str(i) '.exr']);
        exrwrite(normal, [output_dir '/Normal-' num2str(i) '.exr']);
    end
end
disp('done..')
