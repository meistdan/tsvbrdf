function [datapath, result_prefix, width, height, timefile] = load_config_file(configfile)

[filepath, name, ext] = fileparts(configfile);

fid = fopen(configfile);
text = textscan(fid, '%s', 'delimiter', '\n');
text = text{1};

for i = 1:size(text,1)
    line = text(i);
    line = line{1};
    if contains(line, 'Texture Size:')
        tokens = sscanf(line, 'Texture Size: %d x %d');
        height = tokens(1);
        width = tokens(2);
    end
    if contains(line, 'Sample tvBTF')
        tokens = sscanf(line, 'Sample tvBTF%d-%d :');
        result_prefix = ['tvBTF' num2str(tokens(1), '%02d')];
        timefile = [filepath '/' result_prefix '-time.txt'];
        datapath = filepath;        
    end
end
fclose(fid);