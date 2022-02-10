import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier # Extremely randomized trees
from sklearn.model_selection import train_test_split #Data split
from sklearn.utils import shuffle  #Data sequence disorder
from sklearn.model_selection import cross_validate

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
def concat_Fea_grading(points_EMSR_tot1_Features_ROBS,points_EMSR_tot1_Grading,points_EMSR_tot2_Features_ROBS,points_EMSR_tot2_Grading,points_EMSR_tot3_Features_ROBS,points_EMSR_tot3_Grading,points_EMSR_tot4_Features_ROBS,points_EMSR_tot4_Grading,i):
  points_EMSR_tot1_Features_ROBS_data, points_EMSR_tot1_Features_ROBS_other, points_EMSR_tot1_Grading_data, points_EMSR_tot1_Grading_other = train_test_split(points_EMSR_tot1_Features_ROBS[bands].values, points_EMSR_tot1_Grading['grading'].values, test_size=0.9, random_state=i)
  points_EMSR_tot2_Features_ROBS_data, points_EMSR_tot2_Features_ROBS_other, points_EMSR_tot2_Grading_data, points_EMSR_tot2_Grading_other = train_test_split(points_EMSR_tot2_Features_ROBS[bands].values, points_EMSR_tot2_Grading['grading'].values, test_size=0.9, random_state=i)
  points_EMSR_tot3_Features_ROBS_data, points_EMSR_tot3_Features_ROBS_other, points_EMSR_tot3_Grading_data, points_EMSR_tot3_Grading_other = train_test_split(points_EMSR_tot3_Features_ROBS[bands].values, points_EMSR_tot3_Grading['grading'].values, test_size=0.9, random_state=i)
  points_EMSR_tot4_Features_ROBS_data, points_EMSR_tot4_Features_ROBS_other, points_EMSR_tot4_Grading_data, points_EMSR_tot4_Grading_other = train_test_split(points_EMSR_tot4_Features_ROBS[bands].values, points_EMSR_tot4_Grading['grading'].values, test_size=0.9, random_state=i)

  points_EMSR_feat = [pd.DataFrame(points_EMSR_tot1_Features_ROBS_data),pd.DataFrame(points_EMSR_tot2_Features_ROBS_data),pd.DataFrame(points_EMSR_tot3_Features_ROBS_data),pd.DataFrame(points_EMSR_tot4_Features_ROBS_data)]
  points_EMSR_feat_total = pd.concat(points_EMSR_feat)
  points_EMSR_feat_total.columns= bands

  points_EMSR_grading = [pd.DataFrame(points_EMSR_tot1_Grading_data),pd.DataFrame(points_EMSR_tot2_Grading_data),pd.DataFrame(points_EMSR_tot3_Grading_data),pd.DataFrame(points_EMSR_tot4_Grading_data)]
  points_EMSR_grading_total = pd.concat(points_EMSR_grading)
  points_EMSR_grading_total.columns= ['grading']
  return points_EMSR_feat_total,points_EMSR_grading_total

# Function: get the data required for parameter adjustment
def parameter_need_data(i,a=points_EMSR171_tot1_Features_ROBS, aa=points_EMSR171_tot1_Grading, b=points_EMSR171_tot2_Features_ROBS, bb=points_EMSR171_tot2_Grading, c=points_EMSR171_tot3_Features_ROBS, cc=points_EMSR171_tot3_Grading,  d=points_EMSR171_tot4_Features_ROBS,  dd=points_EMSR171_tot4_Grading):
  points_EMSR171_feat_function, points_EMSR171_grad_function = concat_Fea_grading(a,  aa,  b,  bb,  c,  cc,  d,  dd, i)
  points_feat_171_function, points_grad_171_function = shuffle(points_EMSR171_feat_function, points_EMSR171_grad_function, random_state=i)#数据打乱
  points_feat_171_DataFrame_function = pd.DataFrame(points_feat_171_function)
  points_grad_171_DataFrame_function = pd.DataFrame(points_grad_171_function)
  xbands_171_function = points_feat_171_DataFrame_function[bands].values
  yclass_171_function = points_grad_171_DataFrame_function['grading'].values
  return xbands_171_function, yclass_171_function

# Function: get the result of the required parameter min_samples_leaf of Extremely randomized trees
def ETclassifier_CV_get_min_samples_leaf(j,xbands_need, yclass_need):
  print("The following is the result of the% d adjustment min_samples_leaf"%(j))
  score_lt_etc_function = []
  for i in range(1,15):
    etc_function = ExtraTreesClassifier(n_estimators=701, max_features=27, min_samples_split=2, min_samples_leaf=i, random_state=90)
    cv_result_function = cross_validate(etc_function, xbands_need, yclass_need, cv=10)
    test_score_etc_cv = cv_result_function['test_score'].mean()
    time_etc_cv = cv_result_function['fit_time'].mean() + cv_result_function['score_time'].mean()
    print(i,test_score_etc_cv,time_etc_cv*5,'second')
    score_lt_etc_function.append(test_score_etc_cv)
  print('=='*40)
  return score_lt_etc_function

xbands_171_1, yclass_171_1 = parameter_need_data(i=1)
xbands_171_2, yclass_171_2 = parameter_need_data(i=2)
xbands_171_3, yclass_171_3 = parameter_need_data(i=3)
xbands_171_4, yclass_171_4 = parameter_need_data(i=4)
xbands_171_5, yclass_171_5 = parameter_need_data(i=5)
xbands_171_6, yclass_171_6 = parameter_need_data(i=6)
xbands_171_7, yclass_171_7 = parameter_need_data(i=7)
xbands_171_8, yclass_171_8 = parameter_need_data(i=8)
xbands_171_9, yclass_171_9 = parameter_need_data(i=9)
xbands_171_10, yclass_171_10 = parameter_need_data(i=10)

etc_cv_min_samples_leaf_1 = ETclassifier_CV_get_min_samples_leaf(1,xbands_171_1,yclass_171_1)
etc_cv_min_samples_leaf_2 = ETclassifier_CV_get_min_samples_leaf(2,xbands_171_2,yclass_171_2)
etc_cv_min_samples_leaf_3 = ETclassifier_CV_get_min_samples_leaf(3,xbands_171_3,yclass_171_3)
etc_cv_min_samples_leaf_4 = ETclassifier_CV_get_min_samples_leaf(4,xbands_171_4,yclass_171_4)
etc_cv_min_samples_leaf_5 = ETclassifier_CV_get_min_samples_leaf(5,xbands_171_5,yclass_171_5)
etc_cv_min_samples_leaf_6 = ETclassifier_CV_get_min_samples_leaf(6,xbands_171_6,yclass_171_6)
etc_cv_min_samples_leaf_7 = ETclassifier_CV_get_min_samples_leaf(7,xbands_171_7,yclass_171_7)
etc_cv_min_samples_leaf_8 = ETclassifier_CV_get_min_samples_leaf(8,xbands_171_8,yclass_171_8)
etc_cv_min_samples_leaf_9 = ETclassifier_CV_get_min_samples_leaf(9,xbands_171_9,yclass_171_9)
etc_cv_min_samples_leaf_10 = ETclassifier_CV_get_min_samples_leaf(10,xbands_171_10,yclass_171_10)