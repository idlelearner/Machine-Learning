classdef tree < handle
    properties
        isLeafNode = 0;
        feature_no = 0;
        branchOutput = 0;
        childern;
        nChildren = 0;
        FeatureVals;
    end
    
    methods
        
        function n= tree(isLeaf, arg1, arg2)
            if (isLeaf==0)
                n.isLeafNode = 0;
                n.branchOutput = 0;
                n.feature_no = arg1;
                n.nChildren = length(arg2);            
                n.childern = cell(n.nChildren, 1);
                n.FeatureVals = arg2;
            else
                n.isLeafNode = 1;
                n.branchOutput = arg1;
            end   
        end
        
        function addLeafChild(n, fVal, Output)
            in = find(n.FeatureVals == fVal);
            n.childern{in} = tree(1, Output, Output);
        end
        
        function addChild(n, fVal, f, allVals)
            in = find(n.FeatureVals == fVal);
            n.childern{in} = tree(0, f, allVals);
        end
        
        function c = getChild(n, fVal)
            in = find(n.FeatureVals == fVal);
            c = n.childern{in};
        end
        
    end
    
end