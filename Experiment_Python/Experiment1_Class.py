import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.ensemble import ExtraTreesClassifier #Extremely randomized trees
from sklearn.model_selection import train_test_split #Data split
from sklearn.model_selection import StratifiedKFold #Stratified sampling
from sklearn.metrics import confusion_matrix # Confusion matrix
from sklearn.utils import shuffle  #Data sequence disorder
from rotation_forest import RotationForestClassifier #Rotation Forest

#============================================
#             Read dataset
#============================================
# Robustness normalized Feature data
points_EMSR171_tot1_Features_ROBS = pd.read_csv('points_EMSR171_tot1_Features_ROBS.csv')
points_EMSR171_tot2_Features_ROBS = pd.read_csv('points_EMSR171_tot2_Features_ROBS.csv')
points_EMSR171_tot3_Features_ROBS = pd.read_csv('points_EMSR171_tot3_Features_ROBS.csv')
points_EMSR171_tot4_Features_ROBS = pd.read_csv('points_EMSR171_tot4_Features_ROBS.csv')
# labeled data
points_EMSR171_tot1_Grading = pd.read_csv('points_EMSR171_tot1_Grading.csv')
points_EMSR171_tot2_Grading = pd.read_csv('points_EMSR171_tot2_Grading.csv')
points_EMSR171_tot3_Grading = pd.read_csv('points_EMSR171_tot3_Grading.csv')
points_EMSR171_tot4_Grading = pd.read_csv('points_EMSR171_tot4_Grading.csv')

bands = ['BAI',
      'NBR_pre','NBR_post','dNBR',
      'NDVI_pre','NDVI_post','dNDVI',
      'NDWI_pre','NDWI_post','dNDWI',
      'VARI_pre','VARI_post','dVARI',
      'CIre_pre','CIre_post','dCIre',
      'NDVIre1_pre','NDVIre1_post','dNDVIre1',
      'NDVIre2_pre','NDVIre2_post','dNDVIre2'
      ,'MSRre_pre','MSRre_post','dMSRre'
      ,'MSRren_pre','MSRren_post','dMSRren']


#========================================
#         Get 10% of the dataset
#========================================

def concat_Fea_grading(points_EMSR_tot1_Features_ROBS,points_EMSR_tot1_Grading,points_EMSR_tot2_Features_ROBS,points_EMSR_tot2_Grading,points_EMSR_tot3_Features_ROBS,points_EMSR_tot3_Grading,points_EMSR_tot4_Features_ROBS,points_EMSR_tot4_Grading,i,train_size):
  points_EMSR_tot1_Features_ROBS_data, points_EMSR_tot1_Features_ROBS_other, points_EMSR_tot1_Grading_data, points_EMSR_tot1_Grading_other = train_test_split(points_EMSR_tot1_Features_ROBS[bands].values, points_EMSR_tot1_Grading['grading'].values, train_size=train_size, random_state=i)
  points_EMSR_tot2_Features_ROBS_data, points_EMSR_tot2_Features_ROBS_other, points_EMSR_tot2_Grading_data, points_EMSR_tot2_Grading_other = train_test_split(points_EMSR_tot2_Features_ROBS[bands].values, points_EMSR_tot2_Grading['grading'].values, train_size=train_size, random_state=i)
  points_EMSR_tot3_Features_ROBS_data, points_EMSR_tot3_Features_ROBS_other, points_EMSR_tot3_Grading_data, points_EMSR_tot3_Grading_other = train_test_split(points_EMSR_tot3_Features_ROBS[bands].values, points_EMSR_tot3_Grading['grading'].values, train_size=train_size, random_state=i)
  points_EMSR_tot4_Features_ROBS_data, points_EMSR_tot4_Features_ROBS_other, points_EMSR_tot4_Grading_data, points_EMSR_tot4_Grading_other = train_test_split(points_EMSR_tot4_Features_ROBS[bands].values, points_EMSR_tot4_Grading['grading'].values, train_size=train_size, random_state=i)

  points_EMSR_feat = [pd.DataFrame(points_EMSR_tot1_Features_ROBS_data),pd.DataFrame(points_EMSR_tot2_Features_ROBS_data),pd.DataFrame(points_EMSR_tot3_Features_ROBS_data),pd.DataFrame(points_EMSR_tot4_Features_ROBS_data)]
  points_EMSR_feat_total = pd.concat(points_EMSR_feat)
  points_EMSR_feat_total.columns= bands

  points_EMSR_grading = [pd.DataFrame(points_EMSR_tot1_Grading_data),pd.DataFrame(points_EMSR_tot2_Grading_data),pd.DataFrame(points_EMSR_tot3_Grading_data),pd.DataFrame(points_EMSR_tot4_Grading_data)]
  points_EMSR_grading_total = pd.concat(points_EMSR_grading)
  points_EMSR_grading_total.columns= ['grading']
  return points_EMSR_feat_total,points_EMSR_grading_total

# Function: get the data required for parameter adjustment
def parameter_need_data171(i,train_size, a=points_EMSR171_tot1_Features_ROBS, aa=points_EMSR171_tot1_Grading, b=points_EMSR171_tot2_Features_ROBS, bb=points_EMSR171_tot2_Grading, c=points_EMSR171_tot3_Features_ROBS, cc=points_EMSR171_tot3_Grading,  d=points_EMSR171_tot4_Features_ROBS,  dd=points_EMSR171_tot4_Grading):
  points_EMSR171_feat_function, points_EMSR171_grad_function = concat_Fea_grading(a,  aa,  b,  bb,  c,  cc,  d,  dd, i, train_size)
  points_feat_171_function, points_grad_171_function = shuffle(points_EMSR171_feat_function, points_EMSR171_grad_function, random_state=i)#数据打乱
  points_feat_171_DataFrame_function = pd.DataFrame(points_feat_171_function)
  points_grad_171_DataFrame_function = pd.DataFrame(points_grad_171_function)
  xbands_171_function = points_feat_171_DataFrame_function[bands].values
  yclass_171_function = points_grad_171_DataFrame_function['grading'].values
  return xbands_171_function, yclass_171_function


#Classifier definition
rfc_function = RandomForestClassifier(n_estimators=501, max_features=8, min_samples_split=2,min_samples_leaf=1,criterion='gini',random_state=90)
etc_function = ExtraTreesClassifier(n_estimators=701,max_features=27, min_samples_split=2,min_samples_leaf=1,criterion='entropy')
rofc_function = RotationForestClassifier(n_estimators=21,n_features_per_subset=2)

#===========================================================================================
# Classification and writing confusion matrix===============>>>>Extremely randomized trees
#===========================================================================================

with open("EMSR171_etc.csv","w") as csvfile_etc:
  with open("EMSR171_etc_recall.csv","w") as csvfile_recall_etc:
    print("The following is the result of Extremely randomized trees classification in EMSR171 fire")
    for j in range(11,211):
      print(j)
      xbands_171, yclass_171 = parameter_need_data171(i=j, train_size=0.07)
      sk = StratifiedKFold(n_splits=10)
      writer_etc = csv.writer(csvfile_etc)
      writer_recall_etc = csv.writer(csvfile_recall_etc)

      name_171 = "EMSR171-"+str(j)+"th"
      writer_etc.writerow([name_171])
      writer_recall_etc.writerow([name_171])
      writer_etc.writerows([['Completely','Highly','Moderately','Negligible']])

      # Hierarchical cross-validation
      for train_index, test_index in sk.split(xbands_171, yclass_171):
        X_train, X_test = xbands_171[train_index], xbands_171[test_index]
        y_train, y_test = yclass_171[train_index], yclass_171[test_index]

        etc_function.fit(X_train, y_train)
        etc_y_pre = etc_function.predict(X_test)
        etc_confusion_matrix_fu_etc = confusion_matrix(y_test, etc_y_pre, labels=[4,3,2,1])

        #Save confusion matrix
        writer_etc.writerows(etc_confusion_matrix_fu_etc)
        csvfile_etc.write("\n")
        csvfile_etc.flush()