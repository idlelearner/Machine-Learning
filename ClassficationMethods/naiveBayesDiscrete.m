function [ testSetErrorRate, testSetStdDev, trainSetErrorRate, trainSetStdDev ] = naiveBayesDiscrete(data_filename, labels_filename, num_splits, train_percent)
% naiveBayes for document classification

dMatrix = csvread(data_filename);
lMatrix = csvread(labels_filename);

%nMatrix = spconvert(dMatrix);

xMatrix = dMatrix;
sMatrix = [zeros(size(lMatrix,1),1) lMatrix];

%tMatrix = eye(size(dMatrix,1),1);
%sMatrix = vertcat(tMatrix, dMatrix);
for i = 1 : size(lMatrix,1)
    %getallids with docid as line number
    sMatrix(i, 1) = i;
    %dMatrix = vertcat(dMatrix(idx,:),i);
end

%sourceMatrix = csvread(filename);
n = size(sMatrix,1);

%fprintf('%d\n',n);

% shuffle the rows of the source matrix
yMatrix = sMatrix(randperm(size(sMatrix,1)),:);

% find the number of classes given
k = size(unique(sMatrix(:, 2)), 1);

% Get the classes in the given data set
classes = unique(sMatrix(:, 1));

% use the train_percent from the input or use the default
if isempty(train_percent)
    train_percent = [0.05 0.1 0.15 0.2 0.25 0.3];
end

for split_num = 1 : num_splits
    % do the 80-20 split based on the function randperm
    randPermIndices = randperm(n);
    testIndices = randPermIndices( 1 : floor(n / 5) );
    
    % get the numbers that are in randPermIndices but not in testIndices
    trainIndices = setdiff(randPermIndices, testIndices);
    xMatrix80 = xMatrix(trainIndices, :);
    xMatrix20 = xMatrix(testIndices, :);
    
    yMatrix80 = yMatrix(trainIndices, :);
    yMatrix20 = yMatrix(testIndices, :);
  
    
    % train the data based on train_percent and evaluate
    for tP = 1 : size(train_percent', 1)
        trainIdx = floor( train_percent(tP) * size(xMatrix80, 1));
        %xActualTrain = xMatrix80(1 : trainIdx, :);
        yActualTrain = yMatrix80(1 : trainIdx, :);
        tActualSize = size(yActualTrain,1);
        xActualTrain = [];
        for i = 1 :  size(yActualTrain)
            idx = xMatrix(:,1)== yActualTrain(i,1);
            tMat = xMatrix(idx,:);
            xActualTrain = vertcat(xActualTrain, tMat);
        end
        
        % class metrics from train data into different class
        for class = 1 : k
            index = (yActualTrain(:, 2) == class);
            yTMat =  yActualTrain(index,:);
            xActualTrainClass{class} = [];
            for j = 1 : size(yTMat)
                idx = xActualTrain(:,1) == yTMat(j,1);
                xActualTrainClass{class} = vertcat(xActualTrainClass{class},xActualTrain(idx,:));
            end
            %Pc = Nc/N
            pC{class} = size(unique(xActualTrainClass{class}(:,1)),1)/tActualSize;
            %Total Number of words per class
            totWords{class} = sum(xActualTrainClass{class}(:,3));
        end
                
        for class = 1 : k    
            wordct = size(unique(xActualTrainClass{class}(:,2)),1);
            wds = unique(xActualTrainClass{class}(:,2));
            for i = 1 : wordct
                %Wi Word count in class
                wID = wds(i,1);
                %fprintf('%d\n',wID);
                tWIdx = xActualTrainClass{class}(:,2)== wID;                
                wC{class}{wID} = sum(xActualTrainClass{class}(tWIdx,3));
                % Total word count
                % Reduntant calc - remove if possible
                tWIdx = xActualTrain(:,2)== wID;
                %Check the count
                totWClass{wID} = sum(xActualTrain(tWIdx,3));
                %tc{class}{wi} = (tot # of wi in class + 1)/ (tot no of
                %words in all class + tot wi in all class)
                tC{class}{wID} = (wC{class}{wID} + 1)/(totWords{class} + totWClass{wID}); 
            end
        end

        %for doc id find its class
        %Get unique doc id for class
        xActualTest = [];
        for j = 1 :  size(yMatrix20)
                idx = xMatrix(:,1)== yMatrix20(j,1);
                tMat = xMatrix(idx,:);
                xActualTest = vertcat(xActualTest, tMat);
        end
           
        for i = 1 : size(yMatrix20)
           %Get docid matrix
           %fprintf('i=%d\n',i);
           docid = yMatrix20(i,1);
           ids = xActualTest(:,1)== docid;
           docMat = xActualTest(ids,:);
           %fprintf('%d \n',docid);
           for class = 1 : k
               pd{class}{docid} = 1;
               for j = 1 : size(docMat)
                    wi = docMat(j,2);
                    pow = docMat(j,3);
                    %p = tC{class}{wi};
                    %fprintf('%d %d %d %d\n',class, docid,wi,pow);
                    if(class>size(tC,2))
                        p = 1;
                    else
                       if(size(tC{class},2) < wi)
                           p=1;
                       else
                           p = tC{class}{wi};
                      end
                    end
                    %pd{class}{docid} = tc{class}{wordid}^freq
                    pd{class}{docid} = pd{class}{docid} * power(p,pow);
               end
           end
        end
        
        %Find maximum in the pd{class}{docid}
        predClass = zeros(size(yMatrix20));
        for i = 1 : size(yMatrix20)
            maximumVal = 0;
            maxClass = 1;
            docid = yMatrix20(i,1);
            for class = 1 : k
                if(maximumVal < pd{class}{docid})
                    maximumVal = pd{class}{docid};
                    maxClass = class;
                end
            end
            predClass(i,1) = docid;
            predClass(i,2) = maxClass;
        end      
       
       testErrorRate(tP) = sum(sum(yMatrix20 ~= predClass));
    end    
    totalErrorRate(split_num, :) = testErrorRate;
end

% divide the entire error rate by the total number of observations
totalErrorRate = totalErrorRate ./ floor(n ./ 5);

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