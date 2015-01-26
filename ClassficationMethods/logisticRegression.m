function [ testSetErrorRate, testSetSD ] = logisticRegression( data_filename, labels_filename, num_splits, train_percent)

% logistic Regression for multivariate classification
dMatrix = csvread(data_filename);
lMatrix = csvread(labels_filename);

%nMatrix = spconvert(dMatrix);

%tMatrix = eye(size(dMatrix,1),1);
sMatrix = [zeros(size(dMatrix,1),1) dMatrix];
%sMatrix = vertcat(tMatrix, dMatrix);
for i = 1 : size(lMatrix,1)
    %getallids with docid as line number
    idx = (dMatrix(:, 1) == i);
    %dMatrix = vertcat(dMatrix(idx,:),i);
    sMatrix(idx,1) = lMatrix(i,:);
    % fprintf('%d %d\n',i,lMatrix(i,:));
end

%sourceMatrix = csvread(filename);
n = size(sMatrix,1);

% shuffle the rows of the source matrix
nMatrix = sMatrix(randperm(size(sMatrix,1)),:);

% separate the x and y matrices
yMatrix = nMatrix(:, 1);
xMatrix = nMatrix(:, 2:end);


% m - number of records in the dataset
% l - number of features for each record
[m, l] = size(xMatrix);

% find the number of classes given
k = size(unique(sMatrix(:, 1)), 1);

% set numIterations to 50 for convergence of the betaMatrix
numIterations = 25;

% eta
eta = 0.01;

for split_num = 1 : num_splits
    % do the 80-20 split based on the function randperm
    randPermIndices = randperm(m);
    testIndices = randPermIndices( 1 : floor(m ./ 5) );
    
    % get the numbers that are in randPermIndices but not in testIndices
    trainIndices = setdiff(randPermIndices, testIndices);
    xMatrix80 = xMatrix(trainIndices, :);
    xMatrix20 = xMatrix(testIndices, :);
    
    yMatrix80 = yMatrix(trainIndices, :);
    yMatrix20 = yMatrix(testIndices, :);
    
    % train the data based on train_percent and evaluate
    for tP = 1 : size(train_percent', 1)
        trainIdx = floor( train_percent(tP) * size(xMatrix80, 1));
        xActualTrain = xMatrix80(1 : trainIdx, :);
        yActualTrain = yMatrix80(1 : trainIdx, :);
        
        % lenx - number of records in xActualTrain
        [lenx, ~] = size(xActualTrain);
        
        % betaMatrix is l by k
        betaMatrix = zeros(l, k);
        % yEncodedMatrix is length(xActualTrain) by k
        yEncodedMatrix = zeros(lenx, k);
        
        % fill each row of the encodedMatrix acc to it's class
        for i = 1 : k
            % get's the indices for which the class value is 'i'
            idx = (yActualTrain(:, 1) == i);
            % sets the i'th index for all those indices to be 1
            yEncodedMatrix(idx, i) = 1;
        end
        
        % Using gradient descent
        for iterationCount = 1 : numIterations
            % pI must've the same dimensions as yEncodedMatrix
            pI = zeros(lenx, k);
            
            for index = 1 : lenx
                pI(index, :) = exp(xActualTrain(index, :) * betaMatrix);
            end
            % eTot is the column wise sum of pI
            %eTot = sum(pI);
            
            % now compute the actual values of pI
            for index = 1 : lenx
                %pI(i, :) = pI(i, :) ./ eTot;
               pI(i, :) = pI(i, :);
            end
            
            % compute the betaMatrix using the gradient descent method
            betaMatrix = betaMatrix - eta * (xActualTrain' * (pI - yEncodedMatrix));
        end
        
        % calculate the xMatrix20 * betaMatrix and classify
        for idx = 1 : size(xMatrix20, 1)
            [~, yPredictionTestMatrix(idx)] = max(exp((xMatrix20(idx, :) * betaMatrix)));
        end
        
        % calculate the misclassifcation rate for corresponding
        % trainPercent
        testErrorRate(tP) = sum(yMatrix20' ~= yPredictionTestMatrix);
    end
    totalErrorRate(split_num, :) = testErrorRate;
    % TODO: remove this break
    %break;
end

totTime = cputime - startTime;

disp('Total time taken is:');
disp(totTime);


% divide the entire error rate by the total number of observations
totalErrorRate = totalErrorRate ./ floor(m ./ 5);

disp('Train Percent Vector');
disp(train_percent);

disp('Mean Error for different train percentages as in train percent vector');
testSetErrorRate = mean(totalErrorRate);
disp(testSetErrorRate);

disp('Deviation of Error for different train percentages as in train percent vector');
testSetSD = std(totalErrorRate);
disp(testSetSD);

plot(train_percent, testSetErrorRate);

end