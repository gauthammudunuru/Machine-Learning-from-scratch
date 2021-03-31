
x = import_worm_data(4000);

function [worm_data, no_worm_data] = import_worm_data(num_of_pics)

    worm_data = zeros(10201, num_of_pics);
    no_worm_data = zeros(10201, num_of_pics);
    for i = 1:num_of_pics

        filename = sprintf('%s_%d.%s','C:\Users\Student\Desktop\Celegans_Train\1\image', i ,'png');
        [cval] = imread(filename);

        cval_t = cval';
        re_cval = cval_t(:); 

        worm_data(:, i) = re_cval;
    end

    for i = 1:num_of_pics

        filename = sprintf('%s_%d.%s','C:\Users\Student\Desktop\Celegans_Train\0\image', i ,'png');
        [cval] = imread(filename);

        cval_t = cval';
        re_cval = cval_t(:); 

        no_worm_data(:, i) = re_cval;
    end
end



