function myRForest2(filename, mV)
% RF using multi split using M number of random features
B = 100;

augmentedMatrix = importdata(filename);

%Shuffle
augmentedMatrix = augmentedMatrix(randperm(size(augmentedMatrix,1)),:);

Yall = augmentedMatrix(:, 1);
nRecords = length(Yall);

Xall = augmentedMatrix(:, 2 : end);
nFeatures = size(Xall, 2);
classLabels=unique(Yall);

K=10;

sSize = floor(nRecords / K);

trainErrorRate = zeros(1, length(mV));
testErrorRate =  zeros(1, length(mV));

featureSizeIndex = 1;

for M = mV

    trainErrorForKFolds = zeros(K,1);
    testErrorForKFolds = zeros(K,1);

    for i = 1 : K
        fprintf('\nkfold : M : %d %d', i, M);
        %Training Set
        X1 = Xall(1: (i-1)*sSize , :);
        X2 = Xall((i*sSize) + 1: K*sSize, :);
        Y1 = Yall(1: (i-1)*sSize , :);
        Y2 = Yall((i*sSize) + 1: K*sSize, :);

        xTrain = [X1;X2];
        yTrain = [Y1;Y2];


        %Test Set
        xtest = Xall(((i-1)*sSize + 1):(i*sSize) , :);
        ytest = Yall(((i-1)*sSize + 1):(i*sSize) , :);

        nTraining= size(xTrain,1);

        clfr_alpha= zeros(B,1);
        clfr_root = cell(B,1);
        clfr_trainIndices = cell(B,1);

        for classifier_no = 1 : B
            % Bootstrap sampling using randsample
            chosenSamples = randsample(nTraining,nTraining,true);
            XtrainingLocal = xTrain(chosenSamples,:);
            YtrainingLocal = yTrain(chosenSamples);

            selectedFeatures = randsample(nFeatures,M);


            InfoGains = impurityOfParentNode(YtrainingLocal) - impurityOfChildrenNode(XtrainingLocal(:,selectedFeatures), YtrainingLocal);

            % Select from specified features only
            [~,feature_no] = max(InfoGains);

            feature_no = selectedFeatures(feature_no);

            %Split on feature number
            splitting_feature = XtrainingLocal(:,feature_no);
            allVals = unique(splitting_feature);
            cardinality = length(allVals);

            % Layer 1 of the decision tree
            root = tree(0,feature_no,allVals);

            for v  = 1:cardinality
                branchedDataIndices = find(splitting_feature == allVals(v));

                if std(YtrainingLocal(branchedDataIndices)) == 0
                    % Make output node
                    root.addLeafChild(allVals(v), mode(YtrainingLocal(branchedDataIndices)));
                else

                    % 2nd Layer Data
                    Xtraining2 = XtrainingLocal(branchedDataIndices, :);
                    Ytraining2 = YtrainingLocal(branchedDataIndices);

                    selectedFeatures = randsample(nFeatures,M);

                    InfoGains2 = impurityOfParentNode(Ytraining2) - impurityOfChildrenNode(Xtraining2(:,selectedFeatures), Ytraining2);

                    [~, feature_no2] = max(InfoGains2);
                    feature_no2 = selectedFeatures(feature_no2);

                    % Split on another feature number
                    splitting_feature2 = Xtraining2(:,feature_no2);
                    allVals2 = unique(splitting_feature2);
                    cardinality2 = length(allVals2);


                    root.addChild(allVals(v), feature_no2, allVals2);
                    for v2  = 1 : cardinality2
                        branchedDataIndices2 = find(splitting_feature2 == allVals2(v2));
                        root.childern{v}.addLeafChild(allVals2(v2), mode(Ytraining2(branchedDataIndices2)))
                    end
                end

            end
            clfr_alpha(classifier_no) = 1 / B;
            clfr_root{classifier_no} = root;
            clfr_trainIndices{classifier_no} = allVals;
            %fprintf('\nclassifier size : %d : %d', classifier_no, M);
        end


        %Prediction
        yTrainPredicted = zeros(size(yTrain));
        for index = 1 : size(xTrain,1)
            op=0;
            for classifier_no = 1 : B
                cnode = clfr_root{classifier_no};
                while cnode.isLeafNode==0
                    val = xTrain(index, cnode.feature_no);
                    % Choose value closer to what bins appeared before
                    fvals = cnode.FeatureVals;
                    [~, valindex] = min(abs(fvals - val));
                    val = fvals(valindex);
                    cnode = cnode.getChild(val);
                end
                op = op + clfr_alpha(classifier_no) * (cnode.branchOutput);
            end
            [~, labelindex] = min(abs(classLabels - op));
            yTrainPredicted(index) = classLabels(labelindex);
            %fprintf('\ntrain prediction index: %d : %d', index, M);
        end


        testError = sum(yTrain ~= yTrainPredicted);
        trainErrorForKFolds(i) = testError / length(yTrain);

        yTestPredicted = zeros(size(ytest));

        for index = 1 : size(xtest, 1)
            op = 0;
            for classifier_no = 1 : B
                cnode = clfr_root{classifier_no};
                while cnode.isLeafNode == 0
                    val = xtest(index,cnode.feature_no);
                    % Choose value closer to what bins appeared before
                    fvals = cnode.FeatureVals;
                    [~, valindex] = min(abs(fvals - val));
                    val = fvals(valindex);
                    cnode = cnode.getChild(val);
                end
                op = op + clfr_alpha(classifier_no) * (cnode.branchOutput);
            end
            [~, labelindex] = min(abs(classLabels - op));
            yTestPredicted(index) = classLabels(labelindex);
        end

        testError = sum(ytest ~= yTestPredicted);
        testErrorForKFolds(i) = testError/length(ytest);
    end

    fprintf('Stats for feature size %d...\n', M);

    disp('10-fold train error rates: ');
    disp(trainErrorForKFolds');

    disp('10-fold test error rates: ');
    disp(testErrorForKFolds');

    trainErrorRate(featureSizeIndex) = mean(trainErrorForKFolds);
    testErrorRate(featureSizeIndex) =  mean(testErrorForKFolds);

    disp('Mean train error for 10-folds: ');
    disp(trainErrorRate(featureSizeIndex));

    disp('Mean test error for 10-folds: ');
    disp(testErrorRate(featureSizeIndex));

    disp('Standard deviation of train error for 10-folds: ');
    disp(std(trainErrorForKFolds));

    disp('Standard deviation of test error for 10-folds: ');
    disp(std(testErrorForKFolds));

    trainErrorRate(featureSizeIndex) = mean(trainErrorForKFolds);
    testErrorRate(featureSizeIndex) =  mean(testErrorForKFolds);
    featureSizeIndex = featureSizeIndex + 1;

end

disp('Mean train errors for each value of the classifier: ');
disp(trainErrorRate);

disp('Mean test errors for each value of the classifier: ');
disp(testErrorRate);

plot(mV, trainErrorRate);
ylabel('Train Error rate');
xlabel('Feature size');

figure;

plot(mV, testErrorRate);
ylabel('Test Error rate');
xlabel('Feature size');
end





