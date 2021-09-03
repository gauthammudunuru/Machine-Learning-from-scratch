clear; close; clc;
epochs=3000;
rng(100);

hidden_units = 2;
output_units = 2;

input = [-1 1 -1 1;-1 1 1 -1];
[D,N]=size(input);
rho=2;

t = [1 1 0 0; 0 0 1 1];

hidden_weight = rand(hidden_units,D);
output_weight = rand(output_units,hidden_units);

hidden_bias_temp = rand(hidden_units,1);
hidden_bias = zeros(hidden_units, N);

output_bias_temp = rand(output_units,1);
output_bias = zeros(output_units, N);

figure(1);
for i=1:epochs
    
    a1 = hidden_weight*input+hidden_bias;

    z1 = sigmoid(a1);
    
    a2 = output_weight*z1+output_bias;
%     z2 = sigmoid(a2);
    z2 = softmax_activation(a2);
    y=z2;

    Error = mean((y-t).^2,2);
    err(i) = sum(Error);
    semilogy(i,Error,'bo','MarkerFaceColor','b','MarkerSize',1)
    xlim([0 epochs])
    grid on
    hold on

    del_output = y-t;
    del_hidden = output_weight'*del_output.*sigmoid_dervative(z1);
    
    output_weight=output_weight-rho*del_output*z1';
    hidden_weight=hidden_weight-rho*del_hidden*input';
    

    mean_del_hidden = mean(del_hidden,2);
    mean_del_output = mean(del_output,2);
    
    hidden_bias_temp=hidden_bias_temp-rho*mean_del_hidden;
    output_bias_temp=output_bias_temp-rho*mean_del_output;
    
    hidden_bias=repmat(hidden_bias_temp,1,N);
    output_bias=repmat(output_bias_temp,1,N);
    
end

figure();
input = [-1 1 -1 1;-1 1 1 -1];
tar = [0 0 1 1];
t = [1 1 0 0; 0 0 1 1];
x1 = linspace(-3,3,50);
x2 = linspace(-3,3,50);
[p1,p2] = meshgrid(x1,x2);
g1 = p1(:);
g2 = p2(:);
g = [g1 g2];
K = size(g,1);
temp = hidden_bias(:,1);
hidden_bias = repmat(temp,1,K);

temp1 = output_bias(:,1);
output_bias = repmat(temp,1,K);

a1 = hidden_weight*g'+hidden_bias;

z1 = sigmoid(a1);

a2 = output_weight*z1+output_bias;
%     z2 = sigmoid(a2);
z2 = sigmoid(a2);
y=z2;
ga = reshape(y(2,:),size(p1));



plot3(p1,p2, ga,"ColorMode","auto")
hold on
plot3(input(1,:),input(2,:),tar,'bo',"LineWidth",2)



%%
function [result] = sigmoid(x)
    result = 1./(1 + exp(-x));
end


function [z]=softmax_activation(a)
    L = size(a,2);
    for i=1:L
        a1 = a(1,i);
        a2 = a(2,i);
        z(1,i) = exp(a1)/(exp(a1) + exp(a2));
        z(2,i) = exp(a2)/(exp(a1) + exp(a2));
    end
end


function [res] = sigmoid_dervative(x)
    res = x.*(1-x);
end

function [error] = error_function(a,t)
    L = size(a,2);
    error = 0;
    for i=1:L
        y1 = a(1,i);
        y2 = a(2,i);
        t1 = t(1,i);
        t2 = t(2,i);
        error = error + (-t1*log(y1) - t2*log(y2));
    end
end
