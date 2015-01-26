function mysmosvm( filename, numruns )
%SMO Implementation for avg runtime
dMatrix = csvread(filename);
sMatrix = dMatrix(randperm(size(dMatrix,1)),:);
m = size(sMatrix,1);
Y = sMatrix(:,1);
idx = (Y(:, 1) == 3);
Y(idx,1) = 0;
X = sMatrix(:,2:end);

C=0.00001;
tol = 1e-3;

iter = 5;
m = size(X, 1);
n = size(X, 2);
max_p = 100;

Y(Y==0) = -1;
for t = 1:numruns
    
    startTime = cputime;
    % Variables
    alp = zeros(m, 1);
    b = 0;
    E = zeros(m, 1);
    p = 0;
    eta = 0;
    L = 0;
    H = 0;

    % Pre-compute the Kernel Matrix
    K = zeros(m);
        for i = 1:m
            for j = i:m
                K(i,j) = kernal(X(i,:)', X(j,:)');
                K(j,i) = K(i,j); 
            end
        end
    while p < max_p,
        
        ch_alp = 0;
        for i = 1:m,
            E(i) = b + sum (alp.*Y.*K(:,i)) - Y(i);
            
            if ((Y(i)*E(i) < -tol && alp(i) < C) || (Y(i)*E(i) > tol && alp(i) > 0)),
                
                % we select j randomly.
                j = ceil(m * rand());
                while j == i, 
                    j = ceil(m * rand());
                end

                E(j) = b + sum (alp.*Y.*K(:,j)) - Y(j);
                
                % Save old alp
                alp_i = alp(i);
                alp_j = alp(j);

                if (Y(i) == Y(j)),
                    L = max(0, alp(j) + alp(i) - C);
                    H = min(C, alp(j) + alp(i));
                else
                    L = max(0, alp(j) - alp(i));
                    H = min(C, C + alp(j) - alp(i));
                end
                
                if (L == H),
                    continue;
                end
                
                eta = 2 * K(i,j) - K(i,i) - K(j,j);
                if (eta >= 0),
                    continue;
                end

                alp(j) = alp(j) - (Y(j) * (E(i) - E(j))) / eta;

                alp(j) = min (H, alp(j));
                alp(j) = max (L, alp(j));

                if (abs(alp(j) - alp_j) < tol),
                    alp(j) = alp_j;
                    continue;
                end

                alp(i) = alp(i) + Y(i)*Y(j)*(alp_j - alp(j));
                b1 = b - E(i) ...
                    - Y(i) * (alp(i) - alp_i) *  K(i,j)' ...
                    - Y(j) * (alp(j) - alp_j) *  K(i,j)';
                b2 = b - E(j) ...
                    - Y(i) * (alp(i) - alp_i) *  K(i,j)' ...
                    - Y(j) * (alp(j) - alp_j) *  K(j,j)';
                
                % Compute b by (19).
                if (0 < alp(i) && alp(i) < C),
                    b = b1;
                elseif (0 < alp(j) && alp(j) < C),
                    b = b2;
                else
                    b = (b1+b2)/2;
                end
                ch_alp = ch_alp + 1;   
            end
            
        end
        if (ch_alp == 0),
            p = p + 1;
        else
            p = 0;
        end
    end
    timet = cputime - startTime;
    timetak(t) = timet;
end
fprintf('\nMean runtime : %d', mean(timetak))
fprintf('\nSD runtime : %d', std(timetak))
end

function [kv]=kernal(x1,x2)
    ktmp=0;
    s=10;
    [m n]=size(x1);
    for i=1:n
        u=(x1(m,i)-x2(m,i))*(x1(m,i)-x2(m,i));
        ktmp=ktmp+exp((-1*u*0.5)/(s*s));
    end
    kv=ktmp;
end
