function iGVals=impurityOfChildrenNode(features, labels)
nFeatures = size(features, 2);
iGVals = zeros(1, nFeatures);

for f=1 : nFeatures
feature = features(:,f);
%Compute cardinality of feature
vals = unique(feature);
cardinality = length(vals);
N = length(labels);
infoval = 0;

for i = 1 : cardinality 
    branchedDataIndices = find(feature == vals(i));
    Ni=length(branchedDataIndices);
    infoval = infoval + (Ni / N) * impurityOfParentNode(labels(branchedDataIndices));
end
iGVals(f) = infoval;
end
