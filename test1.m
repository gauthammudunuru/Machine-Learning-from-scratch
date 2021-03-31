close; clear; clc;
N = 4000;
[x1, x2] = import_worm_data(N);
X = [x1';x2'];
[coeff,score,latent] = pca(X);
A = score(:,1:100);
A1 = [A ones(8000,1)];
t = [ones(N,1); -ones(N,1)];
weight = logistic(A,t);
y1 = A1*weight;
y2 = sigmoid_activate(y1');
y3 = classify(y2);

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


function [w] = logistic(A,t)
    % A = data(known)
    % t = target(known)
    iter = 100;
    [N,d] = size(A);
    o = ones(N,1);
    x = [A o];
    w = rand(d+1,1);
    R =1e-3;
    J = (sigmoid_activate(x(1,:)*w)-t(1))*x(1,:);
    
    for i =1:iter
        
        for j = 1:N
            J = J + (sigmoid_activate(x(j,:)*w)-t(j))*x(j,:);
        end
        
        w = w-R*J';
        
        if(J <= 0)
            break;
        end  
        
    end

end

function [y] = sigmoid_activate(x_entries)
y = zeros(1,size(x_entries, 2));
for i = 1:size(x_entries, 2)
    answ = 1/(1 + exp(-x_entries(1, i)));
    y(1, i) = answ;
end

end

function [y] = classify(y_hat)

y = zeros(1, size(y_hat, 2));

    for i = 1:size(y_hat, 2)
        if y_hat(1, i) >= 0.5
            y(1, i) = 1;
        end
        if y_hat(1, i) < 0.5
            y(1, i) = 0;
        end
    end
    
end
