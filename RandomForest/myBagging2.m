function myBagging2(filename, bV)
% Bagging using multi split using B number of classifiers

sourceM = importdata(filename);

%shuffle the matrix
sourceM = sourceM(randperm(size(sourceM, 1)), :);

Yall = sourceM(:, 1);
nRecords = length(Yall);

Xall = sourceM(:, 2 : end);
classLabels = unique(Yall);

K = 10;

sSize= floor(nRecords / K);

trainErrorRate = zeros(1, length(bV));
testErrorRate =  zeros(1, length(bV));

classifierIndex = 1;

for B = bV
    trainingErrorForKFolds = zeros(K, 1);
    testErrorforKFolds = zeros(K, 1);

    for i = 1 : K

        %Training Set
        X1 = Xall(1: (i-1)*sSize , :);
        X2 = Xall((i*sSize) + 1: K*sSize, :);
        Y1 = Yall(1: (i-1)*sSize , :);
        Y2 = Yall((i*sSize) + 1: K*sSize, :);

        xTrain = [X1;X2];
        yTrain = [Y1;Y2];


        %Test Set
        xTest = Xall(((i-1)*sSize + 1):(i*sSize) , :);
        yTest = Yall(((i-1)*sSize + 1):(i*sSize) , :);

        xsSize = size(xTrain,1);

        classifier_alpha= zeros(B,1);
        classifier_root = cell(B,1);
        classifier_trainIndices = cell(B,1);

        for classifier_no=1:B

            % bootstrap sampling done using randsample
            chosenSamples = randsample(xsSize, xsSize, true);
            XtrainingLocal = xTrain(chosenSamples,:);
            YtrainingLocal = yTrain(chosenSamples);

            InfoGains = impurityOfParentNode(YtrainingLocal) - impurityOfChildrenNode(XtrainingLocal, YtrainingLocal);

            [~,feature_no] = max(InfoGains);

            %Split on feature number
            splitting_feature = XtrainingLocal(:,feature_no);
            allVals = unique(splitting_feature);
            cardinality = length(allVals);

            root = tree(0,feature_no,allVals);

            % Construct the layer 2 of the tree
            for v  = 1:cardinality
                branchedDataIndices = find(splitting_feature==allVals(v));

                if std(YtrainingLocal(branchedDataIndices)) == 0
                    % Make output node
                    root.addLeafChild(allVals(v), mode(YtrainingLocal(branchedDataIndices)));
                else

                    % 2nd Layer Data
                    Xtraining2=XtrainingLocal(branchedDataIndices,:);
                    Ytraining2=YtrainingLocal(branchedDataIndices);

                    InfoGains2 = impurityOfParentNode(Ytraining2) - impurityOfChildrenNode(Xtraining2, Ytraining2);

                    [~,feature_no2] = max(InfoGains2);

                    % Split on another feature number
                    splitting_feature2 = Xtraining2(:,feature_no2);
                    allVals2 = unique(splitting_feature2);
                    cardinality2 = length(allVals2);


                    root.addChild(allVals(v), feature_no2, allVals2);
                    for v2  = 1:cardinality2
                        branchedDataIndices2 = find(splitting_feature2==allVals2(v2));
                        root.childern{v}.addLeafChild(allVals2(v2), mode(Ytraining2(branchedDataIndices2)))
                    end
                end

            end
            classifier_alpha(classifier_no) = 1/B;
            classifier_root{classifier_no} = root;
            classifier_trainIndices{classifier_no} = allVals;
        end


        %Prediction
        yTrainPredicted = zeros(size(yTrain));
        for index = 1 : size(xTrain,1)
            op = 0;
            for classifier_no = 1 : B
                cnode = classifier_root{classifier_no};
                while cnode.isLeafNode == 0
                    val = xTrain(index, cnode.feature_no);
                    % if we have no values for a bin - use prev bin values
                    fvals = cnode.FeatureVals;
                    [~, valindex] = min(abs(fvals - val));
                    val = fvals(valindex);
                    cnode = cnode.getChild(val);
                end
                op = op + classifier_alpha(classifier_no) * (cnode.branchOutput);
            end
            [~, labelindex] = min(abs(classLabels - op));
            yTrainPredicted(index) = classLabels(labelindex);
        end


        error = sum(yTrain ~= yTrainPredicted);
        trainingErrorForKFolds(i) = error / length(yTrain);


        yTestPredicted = zeros(size(yTest));

        for index = 1 : size(xTest,1)
            op = 0;
            for classifier_no = 1 : B
                cnode = classifier_root{classifier_no};
                while cnode.isLeafNode == 0
                    val = xTest(index, cnode.feature_no);
                    % Choose value closer to what bins appeared before
                    fvals= cnode.FeatureVals;
                    [~, valindex] = min(abs(fvals - val));
                    val = fvals(valindex);
                    cnode = cnode.getChild(val);
                end
                op = op + classifier_alpha(classifier_no)*(cnode.branchOutput);
            end
            [~, labelindex] = min(abs(classLabels - op));
            yTestPredicted(index) = classLabels(labelindex);
        end

        error = sum(yTest ~= yTestPredicted);
        testErrorforKFolds(i) = error / length(yTest);
    end

    fprintf('Stats for classifier size %d...\n', B);

    disp('10-fold train error rates: ');
    disp(trainingErrorForKFolds');

    disp('10-fold test error rates: ');
    disp(testErrorforKFolds');

    trainErrorRate(classifierIndex) = mean(trainingErrorForKFolds);
    testErrorRate(classifierIndex) =  mean(testErrorforKFolds);

    disp('Mean train error for 10-folds: ');
    disp(trainErrorRate(classifierIndex));

    disp('Mean test error for 10-folds: ');
    disp(testErrorRate(classifierIndex));

    disp('Standard deviation of train error for 10-folds: ');
    disp(std(trainingErrorForKFolds));

    disp('Standard deviation of test error for 10-folds: ');
    disp(std(testErrorforKFolds));

    classifierIndex = classifierIndex + 1;

end

disp('Classifier size: ');
disp(bV);

disp('Mean train errors for each value of the classifier as above: ');
disp(trainErrorRate);

disp('Mean test errors for each value of the classifier as above: ');
disp(testErrorRate);

plot(bV, trainErrorRate);
ylabel('Train Error rate');
xlabel('Classifiers size');

figure;

plot(bV, testErrorRate);
ylabel('Test Error rate');
xlabel('Classifiers sSize');

end
