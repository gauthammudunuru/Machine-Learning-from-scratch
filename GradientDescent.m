x = load('carbig.mat','Weight','Horsepower');

data = [x.Weight,x.Horsepower];
data(any(isnan(data), 2), :) = [];
mu1 = mean(data(:,1));
mu2 = mean(data(:,2));
sd1 = std(data(:,1));
sd2 = std(data(:,2));
% data = zscore(data);
weight = (data(:,2)-mu2)/sd2;
horsepower = (data(:,1)-mu1)/sd1;
t = horsepower;
iter =100;
[r,c] = size(weight);
o = ones(r,1);
A = [weight,o];
w = rand(c+1,1);
R =1e-3;
J = 2*(w'*(A'*A)) - 2*(t'*A);
for i =1:iter
    J = 2*(w'*(A'*A)) - 2*(t'*A);
    w = w-R*J';
    if(J <= 0)
        break;
    end    
end
y = A*w*sd2 + mu2;
weight = weight*sd1 + mu1;
hold on
plot(weight,y,'green')
plot(data(:,1),data(:,2),'.r')
hold off
title('Matlab''s "carbig" dataset')
legend('Gradient Descent');
xlabel('Weight');
ylabel('Horsepower');

figure()

y_hat = A*(((inv(A'*A))*A')*horsepower)*sd2 +mu2;

hold on
plot(weight,y_hat,'blue')
plot(data(:,1),data(:,2),'.r')
hold off
title('Matlab''s "carbig" dataset')
legend('Closed Form')
xlabel('Weight');
ylabel('Horsepower');

