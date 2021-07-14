function timeV = load_sample_timefile(timefile)
fileID = fopen(timefile);
timeV = textscan(fileID, '%d %d %d %d','Delimiter',':');
hours = timeV{:,2};
minutes = timeV{:,3};
seconds = timeV{:,4};
for i = 2:size(hours,1)
    if hours(i-1) > hours(i)
        hours(i) = hours(i) + 24;
    end
end
timeV = hours * 60 * 60 + minutes * 60 + seconds;
timeV = cast(timeV, 'double');
timeV = (timeV-min(timeV));
timeV = timeV/max(timeV);
timeV = timeV';
fclose(fileID);
