import pandas as pd
from IPython.display import display, SVG
from graphviz import Source
from sklearn import preprocessing
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer

data = pd.read_csv("computer-sales1.xlsx")
data = pd.DataFrame(data)
valuedata = data.values

header = list(data.columns)
featureList = []
labelList = data["buys_computer"]
for value in valuedata:
    featureDict = {}
    for i in range(4):
        featureDict[header[i]] = value[i + 1]
    featureList.append(featureDict)

vee = DictVectorizer()
dummyX = vee.fit_transform(featureList).toarray()
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
clf = tree.DecisionTreeClassifier(criterion="entropy")

clf = clf.fit(dummyX, dummyY)
print(clf)
graph = Source(
    tree.export_graphviz(clf, feature_names=vee.get_feature_names_out(), out_file=None)
)
# 显示SVG
svg = SVG(graph.pipe(format="svg"))
display(svg)

with open("output_graph.svg", "wb") as f:
    f.write(graph.pipe(format="svg"))
