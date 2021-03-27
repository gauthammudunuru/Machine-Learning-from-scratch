function [w] = logistic(A,t)
    % A = data(known)
    % t = target(known)
    iter = 100000;
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
