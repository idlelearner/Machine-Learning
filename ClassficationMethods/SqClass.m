function [ teErRt, teSD, trErRt, trainingSetSD ] = SqClass( filename, numCrossval )
% SqClass - Least Squares Linear Discriminant

% read the file and created the sM
sM = csvread(filename);
m = size(sM, 1);

% shuffle the rows of the augmented matrix
sMS = sM(randperm(size(sM,1)),:);

% find the number of classes given
k = size(unique(sMS(:, 1)), 1);

% Get the cs in the given data set
cs = unique(sMS(:, 1));

% form the matrices x and y
yM = zeros(m, k);
xM = [sMS(:, 2:end), ones(m, 1)];


for val = 1 : k
    sMS(:, 1)==cs(val);
    idx = (sMS(:, 1) == cs(val));
    yM(idx,val) = 1;
end

testSize = floor(m / numCrossval);

% k-fold cross validation
for i = 1 : numCrossval
    sTeID = (i-1) * testSize + 1;
    eTeID = i*testSize;
    xTest = xM( sTeID : eTeID, :);
    yTest = yM( sTeID : eTeID, :);
    
    if i == 1
        xT = xM(eTeID + 1 : end, :);
    else
        xTFirst = xM( 1 : sTeID - 1, :);
        xT = vertcat(xM(eTeID + 1 : end, :), xTFirst);
    end
    
    if i == 1
        yT = yM(eTeID + 1 : end, :);
    else
        yTFir = yM( 1 : sTeID - 1, :);
        yT = vertcat(yM(eTeID + 1 : end, :), yTFir);
    end
    
    % compute the w component using the closed form solution
    wM = pinv(xT) * yT;
    
    % get the new output matrix for train and test data each time
    nYTr = (wM' * xT')';
    nYTe = (wM' * xTest')';
    
    % get the max indices of nYTr, nYTe, yT and yTest
    nYTr = nYTr';
    [~, nYTrIdx] = max(nYTr);
    
    nYTe = nYTe';
    [~, nYTeIdx] = max(nYTe);
    
    yT = yT';
    [~, yTrIdx] = max(yT);
    
    yTest = yTest';
    [~, yTeIdx] = max(yTest);
    
    % find the total # of misclassifications
    teE(i) = sum(nYTeIdx ~= yTeIdx);
    trE(i) = sum(nYTrIdx ~= yTrIdx);
end

teE = teE / testSize;
trE = trE / (m - testSize);

disp('Mean Test Error Rate:');
teErRt = mean(teE);
disp(teErRt);
disp('Mean Train Error Rate:');
trErRt = mean(trE);
disp(trErRt);

disp('Deviation of the Test Error Rate:');
teSD = std(teE);
disp(teSD);
disp('Deviation of the Train Error Rate:');
trainingSetSD = std(trE);
disp(trainingSetSD);

end

