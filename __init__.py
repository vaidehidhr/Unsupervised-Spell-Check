import numpy as np
import sklearn.cluster
from pip._vendor.pyparsing import Each
import csv

def cleaning(open, irregularList_training, len):
    with open("Clustering_children.csv", 'r') as f:
        with open("Clustering_children_new.csv", 'w') as f1:
            f.readline() # skip header line
            for line in f:
                f1.write(line)
    
    with open("FeatureMatrix.csv", 'r') as f:
        with open("FeatureMatrix_new.csv", 'w') as f1:
            f.readline() # skip header line
            for line in f:
                f1.write(line)
    
    import os
    import csv
#Read both files
    with open('FeatureMatrix_new.csv', 'r') as a:
        reader = csv.reader(a, delimiter=",")
        verbs = list(reader)
    with open('Clustering_children_new.csv', 'r') as b:
        reader = csv.reader(b, delimiter=",")
        clusters = list(reader)
#Write into combine.csv
    if len(verbs) == len(clusters):
        with open('combine.csv', 'w') as f:
#             writer = csv.writer(f,delimiter = ",")
            writer = csv.writer(f, delimiter=',', lineterminator='\n')
            for i in range(0, len(verbs)):
                temp_list = []
                temp_list.extend(verbs[i])
                temp_list.append(clusters[i])
                writer.writerow(temp_list)
    
    csv_file = csv.reader(open('combine.csv', "r"), delimiter=",")
#loop through csv list
    words = [[], [], [], [], [], [], [], [], [], []]
    for row in csv_file:
        #if current rows 2nd value is equal to input, print that row
        if ('jump' == row[0] or 'nod' == row[0] or 'undo' == row[0] or 'decide' == row[0] or 'knit' == row[0] or 
                'rebuil' == row[0] or 'fanc' == row[0] or 'snitch' == row[0] or 'terrif' == row[0] or 
                'smel' == row[0] or 'misla' == row[0] or 'KO' == row[0] or 'inla' == row[0] or 'spen' == row[0] or 
                'rebuil' == row[0] or 'spil' == row[0] or 'red' == row[0] or 'overd' == row[0] or 
                'fle' == row[0] or 'subserv' == row[0] or 'sail' == row[0] or 'sho' == row[0]):
                if ('0' in row[1]):
                    words[0].append(row[0])
                if ('1' in row[1]):
                    words[1].append(row[0])
                if ('2' in row[1]):
                    words[2].append(row[0])
                if ('3' in row[1]):
                    words[3].append(row[0])
                if ('4' in row[1]):
                    words[4].append(row[0])
                if ('5' in row[1]):
                    words[5].append(row[0])
                if ('6' in row[1]):
                    words[6].append(row[0])
                if ('7' in row[1]):
                    words[7].append(row[0])
                if ('8' in row[1]):
                    words[8].append(row[0])
                if ('9' in row[1]):
                    words[9].append(row[0])
    
    for i in range(0, len(words)):
        print(i)
        print(words[i])
    
    with open('irregular_words.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(irregularList_training)
    os.remove('FeatureMatrix.csv')
    os.remove('Clustering_children.csv')

def createClusterDictionary(verbLines_training):

    clusterDictionary_training = dict()
    for x in verbLines_training:
        x=x.replace("\n","")
        splitOnComma= x.split(',')
        key = splitOnComma[0].replace("@", "")
        splitOnComma.pop(0)
        splitOnComma.remove('')
        if (key not in clusterDictionary_training):
            clusterDictionary_training[key] = splitOnComma
    
    return clusterDictionary_training

def extractRules(inflectionsListDictionary):
    ruleList = {}
    for key, value in inflectionsListDictionary.items():
        for str in value:
            if (str in ruleList):
                ruleList[str].append(key)
            else:
                ruleList[str] = [key]
    return ruleList

def extractInflations(clusterDictionary_training):
    inflectionsList_training = {}
    irregularList_training={}
    for key, value in clusterDictionary_training.items():
        tempList = []
        regular=True
        almost_regular=True
        for i in value:
            if key not in i:
                regular=False
                if key[:-1] not in i:
                    almost_regular=False
        if regular==False and almost_regular==False:
            irregularList_training[key]=value
            continue
        if regular==True:
            for str in value:
                str = str.replace(key, "", 1)
                if str!="" and key[-1]==str[0]:
                    str=str.replace(str[0],"D",1)
                if str in tempList:
                    continue
                tempList.append(str)
        else:
            key=key[:-1]
            for str in value:
                str = str.replace(key, "", 1)
                if str in tempList:
                    continue
                tempList.append(str)
        inflectionsList_training[key] = tempList
    return inflectionsList_training, irregularList_training

def createSimilarWordsCluster(inflectionsList_training):
    similarWordsDictionary = {}
    for key, value in inflectionsList_training.items():
        str = key[-3:]
        if (str in similarWordsDictionary):
            similarWordsDictionary[str].append(key)
        else:
            similarWordsDictionary[str] = [key]
    
    return similarWordsDictionary

def extractWords(testFileLines):
    wordList = []
    for x in testFileLines:
        word = x.split('\t')[1]
        wordList.append(word)
    return wordList


def createInputForClustering(clusterDictionary_training, inflectionsList_training, suffixSet):
    count = 0
    with open('FeatureMatrix.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
        writer.writerow(suffixSet)
        for key, value in inflectionsList_training.items():
            count += 1
            writer.writerow([key])
    with open('FeatureMatrix.csv', 'r') as csv_file:
        eachRow = np.zeros([count, len(suffixSet)], dtype = np.int8)
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader)
        rowNum = 0
        for row in reader:
                if row[0] in inflectionsList_training.keys():
                    suffList = inflectionsList_training.get(row[0])
                    for ii in suffList:
                        if ii in header:
                            eachRow[rowNum, header.index(ii)] = 1
                rowNum+=1
        
    with open('FeatureMatrix_number.csv', 'w') as f:
        write_outfile = csv.writer(f, lineterminator='\n')
        write_outfile.writerow(header)
        for i in range(len(eachRow)):
            write_outfile.writerow(eachRow[i])
    return header
            
def getSuffixList(inflectionsList_training):
    suffixlist = []
    for i in inflectionsList_training.values():
        for items in i:
            suffixlist.append(items)
    suffixSet = set(suffixlist)

    return suffixSet

def getClusterLabels(header, clusetring):
    with open('Clustering_children.csv', 'w') as f:
        write_outfile = csv.writer(f, lineterminator='\n')
        write_outfile.writerow(header)
        for i in clusetring.labels_:
            write_outfile.writerow([str(i)])

def clusterRegularVerbs_AgglomerativeClustering():
    from sklearn.cluster import AgglomerativeClustering
    from sklearn import metrics
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    from numpy import genfromtxt
    
    my_data = genfromtxt('FeatureMatrix_number.csv', delimiter=',', skip_header=1)
    clustering = AgglomerativeClustering(linkage='complete', n_clusters=11)
    clustering.fit(my_data)
    params = clustering.get_params(deep=True)
    return clustering
    
def getLemma(lines):
    actualLemmas = []
    for i in lines:
        correctItems = []
        count = 0
        splitForComma = i.split(',')
        for item in splitForComma:
            currentItem = item.replace('@', '')
            correctItems.append(currentItem)
        correctItems.remove('\n')
        probableLemma = min(correctItems, key=len)
        
        for item in correctItems:
            if probableLemma in item:
                count += 1
        if count >= 3:
            actualLemma = probableLemma
        else:
            if count < 2:
                for item in correctItems:
                    if probableLemma[:-1] in item:
                        count += 1
                if count >= 3:
                    actualLemma = probableLemma   
        actualLemmas.append(actualLemma)
   
    rownumber = 0
    newLines = []
    for i in lines:
        i = i.replace(actualLemmas[rownumber]+'@', actualLemmas[rownumber])
        rownumber += 1
        newLines.append(i)
    return newLines
    
if __name__=="__main__": 
    
    verbFile_training= open("celex-verb.list",'r')
    lines=verbFile_training.readlines()
    verbFile_training.close()
    verbLines_training = getLemma(lines)
    clusterDictionary_training = createClusterDictionary(verbLines_training) 
    inflectionsList_training, irregularList_training = extractInflations(clusterDictionary_training) 
    
    verbFile_test= open("testcelex.txt",'r')
    lines_test=verbFile_test.readlines()
    verbFile_test.close()
    verbLines_test = getLemma(lines_test)
    clusterDictionary_test = createClusterDictionary(verbLines_test) 
    inflectionsList_test, irregularList_test = extractInflations(clusterDictionary_test) 
    
    suffixSet= getSuffixList(inflectionsList_training)
    header = createInputForClustering(clusterDictionary_training, inflectionsList_training, suffixSet)
    db=clusterRegularVerbs_AgglomerativeClustering()
    getClusterLabels(header, db)  
    cleaning(open, irregularList_training, len)
    
#     findClusterForTestData()