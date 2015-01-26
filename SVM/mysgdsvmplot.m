function mysgdsvmplot(filename, k1, numruns)
%Pegasos algorithm
%Code for plotting
dMatrix = csvread(filename);
sMatrix = dMatrix(randperm(size(dMatrix,1)),:);
m = size(sMatrix,1);
Y = sMatrix(:,1);
idx = (Y(:, 1) == 3);
Y(idx,1) = -1;
X = sMatrix(:,2:end);
lda = 1;
tot = 0.001;
mItr = [1,50,100,200,1000,10000];
color = ['r' 'k' 'g' 'b' 'm' 'c'];
[N,d]=size(X);

hold on
for i = 1:size(mItr,2)
    for j=1:numruns
        k=k1(j);
        %k=k1;
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
        wT=mean(w,1);
        b=(Y-X*wT');
        objfn = 0;
        for ix=1:size(X,1)
            objfn = objfn + (lda/2)*(norm(wT))^2 + (1/2000)*max(0,1 - Y(ix,1)*dot(wT,X(ix,:)));
        end
        obj(j) = objfn;
        kt(j) = k;
    end
    plot(mItr,obj,color(i))
end
xlabel('#Iterations')
ylabel('Obj Function')
title('#Iterations Vs Obj Function')
legend('k=1','k=20','k=100','k=200','k=1000','k=2000')
hold off
end



