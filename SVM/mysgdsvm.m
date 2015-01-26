function mysgdsvm(filename, k, numruns)
%Pegasos Algorithm
%Calculating average runtime
dMatrix = csvread(filename);
sMatrix = dMatrix(randperm(size(dMatrix,1)),:);
m = size(sMatrix,1);
Y = sMatrix(:,1);
idx = (Y(:, 1) == 3);
Y(idx,1) = -1;
X = sMatrix(:,2:end);
lda = 1;
tot = 0.001;
mItr = [10,50,100, 500,1000,10000];
for i = 1:size(mItr,2)
    for j=1:numruns
        startTime = cputime;
        w=rand(1,size(X,2));
        w=w/(sqrt(lda)*norm(w));
        for t=1:mItr(i)
            b=mean(Y-X*w(t,:)');
            rPerm = randperm(m);
            idx = rPerm(1:k);
            At=X(idx,:);
            yt=Y(idx,:);
            idx1=(At*w(t,:)'+b).*yt<1;
            etat=1/(lda*t);
            w1=(1-etat*lda)*w(t,:)+(etat/k)*sum(At(idx1,:).*repmat(yt(idx1,:),1,size(At,2)),1);
            w(t+1,:)=min(1,1/(sqrt(lda)*norm(w1)))*w1;
            if(norm(w(t+1,:)-w(t,:)) < tot)
                break;
            end
        end
        totTime = cputime - startTime;
        timetaken(j) = totTime;
        Wfunc(j) = norm(w);
        
    end
    fprintf('Iteration=%d',mItr(i));
    fprintf('\nMean time taken = %.4f %%',mean(timetaken));
    fprintf('\nStd Dev time taken = %.4f %%\n',std(timetaken));
end
end

