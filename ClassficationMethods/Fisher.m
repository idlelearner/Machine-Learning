function [ teEr, teSD, trEr, trSD ] = fisher( filename, numCrossval )
% Classfication of Fisher linear discriminant using within class covariance
% between classes calculated from the data given

% Read the file and form the matrix
sM = csvread(filename);
n = size(sM,1);

% shuffle the rows of the source matrix
dM = sM(randperm(size(sM,1)),:);

% find the number of cs given
k = size(unique(dM(:, 1)), 1);

% Get the cs in the given data set
cs = unique(dM(:, 1));

% reduction of dimension
D = k - 1;

% form the matrices x and y
yM = dM(:, 1);
xM = dM(:, 2:end);

t = floor(n / numCrossval);

% k-fold cross validation
for i = 1 : numCrossval
    sTeID = (i-1) * t + 1;
    eTeID = i*t;
    xTest = xM( sTeID : eTeID, :);
    yTe = yM( sTeID : eTeID, :);
    
    if i == 1
        xT = xM(eTeID + 1 : end, :);
    else
        xTrFir = xM( 1 : sTeID - 1, :);
        xT = vertcat(xM(eTeID + 1 : end, :), xTrFir);
    end
    
    if i == 1
        yTr = yM(eTeID + 1 : end, :);
    else
        yTrFir = yM( 1 : sTeID - 1, :);
        yTr = vertcat(yM(eTeID + 1 : end, :), yTrFir);
    end
    
    % calculate the mean of the entire training set
    meanTr = mean(xT);
    
    % init xTr{val} to the dimension of xT(idx, :)
    for c = 1 : size(cs)
        idx = (yTr(:, 1) == cs(c)); % returns all the indices which have 'c'
        % form the c using this
        xTr{c} = xT(idx, :);
    end
    
    % mean of each c
    for c = 1 : size(cs)
        mC{c} = mean(xTr{c});
    end
    
    %print mC;
    % calculate the between-c co-variance for the entire training set
    sB = zeros(size((mC{1} - meanTr)' * (mC{1} - meanTr)));
    for c = 1 : k
        sB = ((mC{c} - meanTr)' * (mC{c} - meanTr)) * size(xTr{c}, 1) + sB;
    end
    
    % within c co-variance
    sW = zeros(size(cov(xTr{1})));
    for c = 1 : k
        sW = sW + cov(xTr{c});
    end
    
    % eigs returns the eigen values and vectors in the descending order
    [eigVector, eigVal] = eigs(pinv(sW)*sB);
    
    wMatrix = eigVector(:, 1 : D);
    
    % calculate xTrP and xTeP vectors using wMatrix
    xTrP = xT * wMatrix;
    xTeP = xTest * wMatrix;

    % init xTrPrC{val} to the dimension of xT(idx, :)
    for c = 1 : k
        idx = (yTr(:, 1) == cs(c)); % returns all the indices which have 'c'
        % form the c using this
        xTrPrC{c} = xTrP(idx, :);
        xTrPrCSize{c} = size(xTrPrC{c}, 1);
    end
    
    % calculate mean and variance for each c 
    for c = 1 : k
        xTrPrMean{c} = mean(xTrPrC{c});
        xTrPrVariance{c} = cov(xTrPrC{c});
    end
    
    % calculate the yInitialPrediction for train data for each c
    for c = 1 : k
        for idx = 1 : size(xTrP, 1)
            %fprintf('%d:%d\n',idx, c);
            yInPredTr{c}(idx) = mvnpdf(xTrP(idx, :), xTrPrMean{c}, xTrPrVariance{c});
        end
    end
    
    % calculate the yInitialPrediction for test data for each c
    for c = 1 : k
        for idx = 1 : size(xTeP, 1)
            yInPredTe{c}(idx) = mvnpdf(xTeP(idx, :), xTrPrMean{c}, xTrPrVariance{c});
        end
    end
    
    % calculate the prediction for Train data
    for idx = 1 : size(yInPredTr{1}', 1)
        maxInd = 1;
        maxV = -1;
        for c = 1 : k
            if yInPredTr{c}(1, idx) > maxV
                maxV = yInPredTr{c}(1, idx);
                maxInd = c;
            end
        end
        yAcPredTr(idx) = cs(maxInd);
    end
    
    % calculate the prediction for Test data
    for idx = 1 : size(yInPredTe{1}', 1)
        maxInd = 1;
        maxV = -1;
        for c = 1 : k
            if yInPredTe{c}(1, idx) > maxV
                maxV = yInPredTe{c}(1, idx);
                maxInd = c;
            end
        end
        yAcPredTe(idx) = cs(maxInd);
    end
    
    % calculates the # of misclassifications
    teErr(i) = sum(yTe' ~= yAcPredTe);
    trErr(i) = sum(yTr' ~= yAcPredTr);
end

teErr = teErr / t;
trErr = trErr / (n - t);

disp('Mean Test Error Rate:');
teEr = mean(teErr);
disp(teEr);
disp('Mean Train Error Rate:');
trEr = mean(trErr);
disp(trEr);

disp('Deviation of the Test Error Rate:');
teSD = std(teErr);
disp(teSD);
disp('Deviation of the Train Error Rate:');
trSD = std(trErr);
disp(trSD);

end


