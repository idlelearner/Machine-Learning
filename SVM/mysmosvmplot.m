function mysmosvmplot( filename, numruns )
%SMO Implementation plotting the obejctive function
dMatrix = csvread(filename);
sMatrix = dMatrix(randperm(size(dMatrix,1)),:);
m = size(sMatrix,1);
Y = sMatrix(:,1);
idx = (Y(:, 1) == 3);
Y(idx,1) = 0;
X = sMatrix(:,2:end);

C=0.00001;
tol = 1e-3;

% Data parameters
m = size(X, 1);
n = size(X, 2);
mp = [1 5 10 50];
colors = ['r' 'k' 'g' 'b' 'm'];
Y(Y==0) = -1;
figure;
hold on;
for t = 1:numruns
    for ix = 1: size(mp,2)
        a = zeros(m, 1);
        b = 0;
        E = zeros(m, 1);
        p = 0;
        eta = 0;
        L = 0;
        H = 0;
        %calculte the Kernel function
        K = zeros(m);
        for i = 1:m
            for j = i:m
                K(i,j) = kernel(X(i,:)', X(j,:)');
                K(j,i) = K(i,j); %the matrix is symmetric
            end
        end

        while p < mp(ix),
            numc = 0;
            for i = 1:m,
                E(i) = b + sum (a.*Y.*K(:,i)) - Y(i);
                if ((Y(i)*E(i) < -tol && a(i) < C) || (Y(i)*E(i) > tol && a(i) > 0)),
                    j = ceil(m * rand());
                    while j == i,
                        j = ceil(m * rand());
                    end
                    
                    E(j) = b + sum (a.*Y.*K(:,j)) - Y(j);
                    
                    % Save old a
                    a_i = a(i);
                    a_j = a(j);
                    
                    if (Y(i) == Y(j)),
                        L = max(0, a(j) + a(i) - C);
                        H = min(C, a(j) + a(i));
                    else
                        L = max(0, a(j) - a(i));
                        H = min(C, C + a(j) - a(i));
                    end
                    
                    if (L == H),
                        continue;
                    end
                    
                    eta = 2 * K(i,j) - K(i,i) - K(j,j);
                    if (eta >= 0),
                        continue;
                    end
                    
                    a(j) = a(j) - (Y(j) * (E(i) - E(j))) / eta;
                    
                    a(j) = min (H, a(j));
                    a(j) = max (L, a(j));
                    
                    if (abs(a(j) - a_j) < tol),
                        a(j) = a_j;
                        continue;
                    end
                    
                    a(i) = a(i) + Y(i)*Y(j)*(a_j - a(j));
                    b1 = b - E(i) ...
                        - Y(i) * (a(i) - a_i) *  K(i,j)' ...
                        - Y(j) * (a(j) - a_j) *  K(i,j)';
                    b2 = b - E(j) ...
                        - Y(i) * (a(i) - a_i) *  K(i,j)' ...
                        - Y(j) * (a(j) - a_j) *  K(j,j)';
                    
                    if (0 < a(i) && a(i) < C),
                        b = b1;
                    elseif (0 < a(j) && a(j) < C),
                        b = b2;
                    else
                        b = (b1+b2)/2;
                    end
                    
                    numc = numc + 1;
                    
                end
                
            end
            
            if (numc == 0),
                p = p + 1;
            else
                p = 0;
            end
        end
        
        idx = a > 0;
        r.X= X(idx,:);
        r.y= Y(idx);
        r.b= b;
        r.a= a(idx);
        r.w = ((a.*Y)'*X)';
        
        % calculate the objective function value
        objfn = 0;
        for i = 1:size(r.a,1)
            for j = 1:size(r.a,1)
                objfn = objfn + r.a(i) - (1/2)*r.y(i)*r.y(j)*K(i,j)*r.a(i)*r.a(j);
            end
        end
        obj_mat(ix) = objfn;
        
    end
    plot(mp,obj_mat,colors(t))
    xlabel('#iterations')
    ylabel('Obj Fn')
    title('#Iterations Vs Obj Fn')
    
end
legend('1','2','3','4','5')
hold off;
end

function [kv]=kernel(x1,x2)
k=0;
s=10;
[m n]=size(x1);
for i=1:n
    u=(x1(m,i)-x2(m,i))*(x1(m,i)-x2(m,i));
    k=k+exp((-1*u*0.5)/(s*s));
end
kv=k;
end