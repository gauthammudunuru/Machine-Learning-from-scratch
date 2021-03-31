clear;close all;
time = zeros(2,20);
k_t = zeros(1,20);
for k = 1:20
    
    rng(100);
    class1=mvnrnd([1 3],[1 0; 0 1],60*k);
    class2=mvnrnd([4 1],[2 0; 0 2],40*k);
    s1 = size(class1);
    s2 = size(class2);
    y = [ones(s1(1),1); -ones(s2(1),1)];
    data = [class1;class2];
    
    [N,l] = size(data);
    k_t(k) = N;
    c = 0.1;
    yy = y*y';
    a = data*data';
    H = (yy.*a);
    f = -ones(N,1);
    A = [-eye(N); eye(N)];
    b = [zeros(N,1); c*ones(N,1)];
    Aeq = y';
    beq = zeros(1);
    
    for i = 1:40
        
        t1 = tic;
        lambda = quadprog(H,f,A,b,Aeq,beq);
        tt = toc(t1);
        time(1,k) = time(1,k) + tt;
        
        t3 =tic;
        lambda1 = fitcsvm(data,y);
        tt2 = toc(t3);
        time(2,k) = time(2,k) + tt2;
    end
    time(1,k) = time(1,k)/40;
    time(2,k) = time(2,k)/40;
    
end
figure();
title(['C = ',num2str(c),'; Computational Efficiency - Quadprog vs SMO'])
hold on

plot(k_t, time(1,:),'red')
plot(k_t, time(2,:),'blue')
xlabel('Number of training samples');
ylabel('Execution time in sec');
legend('Quadprog','SMO',"Location","northwest");
