import time
import os
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import MachinlearingAlgoriths as ML
import PSO as OPTIMIZER



#=========================================  New Featuer selection  ================================
##########################################################################################
##########################################################################################


def hilbert_FS (train, per_vec_len,per_features):     
     templist = [] # to collect vectors 
     tempVect = [] # to collect vector values 
     Features_Weight= [] # collect the final result 
     V_len = int (per_vec_len  * len (train ))
     No_Features = len (train[0])
     selected_featurs= int(per_features* No_Features)
     
     for i in range( No_Features) :  # -- loop for all features 
         k=0
         tempVect.clear()
         for j in range(len(train)) :#-------- loop to collect  vectors 
             tempVect.append(train[j][i])
             k=k+1
             if k == V_len : 
                 k=0 
                 templist.append(tempVect)
                 tempVect=[]
         #print (templist, "\n --------- len of vect 1 >> ",len (templist[1]))        
         No_vect = len (templist)
         #print ("No of vector > ",No_vect)
         #print ("temporary list ",templist)
        
         SumDist = 0
         for m in range ( No_vect): 
             for h in range ( No_vect):
                 if m != h :
                     z= templist[m]
                     x= templist[h]
              
                     SumDist = SumDist+ ((np.dot(x, z) - ( np.max(z))) / ( np.linalg.norm(x) * np.linalg.norm(z))) # templist[m][f] - templist[h][f]
                      
         Features_Weight.append(SumDist) 
         templist.clear()
         
     
     numpyweight= np.array(Features_Weight)
     #MaxWaitFeatures = numpyweight.argsort()[:selected_featurs]    ######## for lowest distance
     MaxWaitFeatures = numpyweight.argsort()[-selected_featurs:]    ########  for highest destance 
     print ( " No of features >> ",selected_featurs )
     print ( " No of Values of Block Size  >> ",V_len )
     
     return MaxWaitFeatures
     
#=============================================================================================
#==================================          Get Data               ==========================
#=============================================================================================

path = "d:/EEG/"
classes = os.listdir(path)


def GetDataClass(path,classLabel):
    data = list()
    files = os.listdir(path)
    for file in files:
        with open(path + '/' + str( file)) as reader:
            attributes = reader.readlines()
        attributes.append(classLabel)
        data.append(attributes)
    return np.array(data)



result=list()
for classLabel in classes:
        data_class = GetDataClass(path + str(classLabel),str(classLabel))
        result.extend(data_class)
data =  np.array(result)
"""
#Scaling data using Z_score_scalar
Z_score_scalar = StandardScaler()
scaled_data = Z_score_scalar.fit_transform(data[:,0:-1])
# Splitting data 
X_train, X_test, y_train, y_test = train_test_split(scaled_data,data[:,-1],test_size=0.1, random_state=0)
"""



# ============================================   get X  and y =====================================================
# ==========================================1s = 179  -  5s = 868  -  10s=1736  -  15s = 2604
Y=list ()
XL= list()
for i in range(len(data)) :
        Y.append(data[i,4097])    
        temp=list()
        for j in range (2604):
            temp.append(int(data[i][j]))
        XL.append(temp)   


#################################################### call  feature selection  ########################################
############################################################################################################
listbestAcc=[]

 
FSPer= 0.20

ML_A= [ "SVM","KNN","DT","NB","RF"]
T = 0
blok_sizes=[0.05,0.10,0.15,0.20,0.30,0.40]
for ml_algorithm in ML_A :
    Acc=list()
    print  ( ml_algorithm )
    for current_BS in blok_sizes :
        NewFeatures = hilbert_FS(XL,current_BS,FSPer)
        XLFS= []
        for i in range(len (XL)):
            temp=[]
            for j in range (len (XL[i])):
                if j in NewFeatures :
                    temp.append(XL[i][j])
            XLFS.append(temp)
        X = np.array(XLFS)  
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
        ML_performance = ML.MachinlearingAlgoriths(traindata=X_train,
                                            testdata=X_test,
                                            trainclass=y_train,
                                            testclass=y_test).ML_callfunction(fun=ml_algorithm) 
        Acc.append(ML_performance[0][0])
        print(ML_performance[0][0]) 
    print ("\n:list of Acc is : \n" , Acc )
    NPacc = np.array (Acc)
    BSMI=NPacc.argsort()[-1:]
    bestAcc = max(Acc)
    best_BS = blok_sizes[BSMI[0]]
    print ( "\n best Accurcy of ",ml_algorithm, " is : " , bestAcc, "  - BS is : ", best_BS)
    if T == 0 :
        listbestAcc.append(list(set(y_test)))
        T = 1 
    listbestAcc.append( bestAcc )
    listbestAcc.append(  best_BS )
print ("----------------------------------------------")
print ("\n",listbestAcc)
with open ('Best-accuracy_HFS.csv', 'a') as appendobj:
    append=csv.writer(appendobj)
    append.writerow(listbestAcc) 

############################################################################################################
############################################################################################################



