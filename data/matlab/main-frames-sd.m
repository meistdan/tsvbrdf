clc;
close all;
clear all;

cd 'D:/projects/MATLAB/tsvbrdf';
addpath './HDRITools/matlab';

frames = 51;

%for s = {'AgedChrome', 'CanyonRock', 'CottonFabric', 'DryMud', 'DuctTape', 'FractureAsphalt', 'GoldFlake', 'Grass', 'Lemon', 'Metal', 'OldWood', 'SnowyGround', 'StoneDirt', 'SyntheticFoam', 'WhiskeyBarrelOak', 'WoodLog' }
for s = {'OldWood', 'SnowyGround', 'StoneDirt', 'SyntheticFoam', 'WhiskeyBarrelOak', 'WoodLog' }
    sample = s{1}
    data_path = ['./data-sd/' sample];
    input_dir = [data_path '/export'];
    output_dir = [data_path '/frames'];
    rmdir(output_dir, 's');
    mkdir(output_dir);
    for i = 0:50
        % input
        albedo = imread([input_dir '/Base Color-' num2str(i) '.png']);
        albedo = single(albedo) / single(intmax(class(albedo)));
        roughness = imread([input_dir '/Roughness-' num2str(i) '.png']);
        roughness = single(roughness) / single(intmax(class(roughness)));
        metallic = imread([input_dir '/Metallic-' num2str(i) '.png']);
        metallic = single(metallic) / single(intmax(class(metallic)));
        if size(metallic,3) == 1
            metallic = repmat(metallic, [1, 1, 3]);
        end
        % output
        colorSpaceDielectricSpecRgb = 0.04;
        colorSpaceDielectricSpecA = 1.0 - 0.04;
		oneMinusReflectivity = colorSpaceDielectricSpecA * (1.0 - metallic);
        diffuse = albedo .* oneMinusReflectivity;
        specular = (1.0 - metallic) .* colorSpaceDielectricSpecRgb + metallic .* albedo;
        exrwrite(diffuse, [output_dir '/Diffuse-' num2str(i) '.exr']);
        exrwrite(specular, [output_dir '/Specular-' num2str(i) '.exr']);
        exrwrite(roughness, [output_dir '/Roughness-' num2str(i) '.exr']);
    end
end
disp('done..')
