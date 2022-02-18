import pandas as pd
import csv
from sklearn.ensemble import ExtraTreesClassifier  # Extremely randomized trees
from sklearn.model_selection import train_test_split  # Data split
from sklearn.model_selection import StratifiedKFold  # Stratified sampling
from sklearn.metrics import confusion_matrix  # Confusion matrix
from sklearn.utils import shuffle  # Data sequence disorder

# ==========================================================================
#             Read dataset
# ==========================================================================
points_EMSR171_tot1_Features_ROBS = pd.read_csv('points_EMSR171_tot1_Features_ROBS.csv')
points_EMSR171_tot2_Features_ROBS = pd.read_csv('points_EMSR171_tot2_Features_ROBS.csv')
points_EMSR171_tot3_Features_ROBS = pd.read_csv('points_EMSR171_tot3_Features_ROBS.csv')
points_EMSR171_tot4_Features_ROBS = pd.read_csv('points_EMSR171_tot4_Features_ROBS.csv')
points_EMSR171_tot1_Grading = pd.read_csv('points_EMSR171_tot1_Grading.csv')
points_EMSR171_tot2_Grading = pd.read_csv('points_EMSR171_tot2_Grading.csv')
points_EMSR171_tot3_Grading = pd.read_csv('points_EMSR171_tot3_Grading.csv')
points_EMSR171_tot4_Grading = pd.read_csv('points_EMSR171_tot4_Grading.csv')

bands = ['BAI',
         'NBR_pre', 'NBR_post', 'dNBR',
         'NDVI_pre', 'NDVI_post', 'dNDVI',
         'NDWI_pre', 'NDWI_post', 'dNDWI',
         'VARI_pre', 'VARI_post', 'dVARI',
         'CIre_pre', 'CIre_post', 'dCIre',
         'NDVIre1_pre', 'NDVIre1_post', 'dNDVIre1',
         'NDVIre2_pre', 'NDVIre2_post', 'dNDVIre2',
         'MSRre_pre', 'MSRre_post', 'dMSRre',
         'MSRren_pre', 'MSRren_post', 'dMSRren']

# ==========================================================================
#             Obtain the data required for the experiment
# ==========================================================================
def concat_Fea_grading_EX3(bands_EX3, points_EMSR_tot1_Features_ROBS, points_EMSR_tot1_Grading,
                           points_EMSR_tot2_Features_ROBS, points_EMSR_tot2_Grading, points_EMSR_tot3_Features_ROBS,
                           points_EMSR_tot3_Grading, points_EMSR_tot4_Features_ROBS, points_EMSR_tot4_Grading, i,
                           train_size):
    points_EMSR_tot1_Features_ROBS_data, points_EMSR_tot1_Features_ROBS_other, points_EMSR_tot1_Grading_data, points_EMSR_tot1_Grading_other = train_test_split(
        points_EMSR_tot1_Features_ROBS[bands_EX3].values, points_EMSR_tot1_Grading['grading'].values,
        train_size=train_size, random_state=i)
    points_EMSR_tot2_Features_ROBS_data, points_EMSR_tot2_Features_ROBS_other, points_EMSR_tot2_Grading_data, points_EMSR_tot2_Grading_other = train_test_split(
        points_EMSR_tot2_Features_ROBS[bands_EX3].values, points_EMSR_tot2_Grading['grading'].values,
        train_size=train_size, random_state=i)
    points_EMSR_tot3_Features_ROBS_data, points_EMSR_tot3_Features_ROBS_other, points_EMSR_tot3_Grading_data, points_EMSR_tot3_Grading_other = train_test_split(
        points_EMSR_tot3_Features_ROBS[bands_EX3].values, points_EMSR_tot3_Grading['grading'].values,
        train_size=train_size, random_state=i)
    points_EMSR_tot4_Features_ROBS_data, points_EMSR_tot4_Features_ROBS_other, points_EMSR_tot4_Grading_data, points_EMSR_tot4_Grading_other = train_test_split(
        points_EMSR_tot4_Features_ROBS[bands_EX3].values, points_EMSR_tot4_Grading['grading'].values,
        train_size=train_size, random_state=i)

    points_EMSR_feat = [pd.DataFrame(points_EMSR_tot1_Features_ROBS_data),
                        pd.DataFrame(points_EMSR_tot2_Features_ROBS_data),
                        pd.DataFrame(points_EMSR_tot3_Features_ROBS_data),
                        pd.DataFrame(points_EMSR_tot4_Features_ROBS_data)]
    points_EMSR_feat_total = pd.concat(points_EMSR_feat)
    points_EMSR_feat_total.columns = bands_EX3

    points_EMSR_grading = [pd.DataFrame(points_EMSR_tot1_Grading_data), pd.DataFrame(points_EMSR_tot2_Grading_data),
                           pd.DataFrame(points_EMSR_tot3_Grading_data), pd.DataFrame(points_EMSR_tot4_Grading_data)]
    points_EMSR_grading_total = pd.concat(points_EMSR_grading)
    points_EMSR_grading_total.columns = ['grading']
    return points_EMSR_feat_total, points_EMSR_grading_total


# Function: get the data
def parameter_need_data171_EX3(i, train_size, bands_EX3, a=points_EMSR171_tot1_Features_ROBS,
                               aa=points_EMSR171_tot1_Grading, b=points_EMSR171_tot2_Features_ROBS,
                               bb=points_EMSR171_tot2_Grading, c=points_EMSR171_tot3_Features_ROBS,
                               cc=points_EMSR171_tot3_Grading, d=points_EMSR171_tot4_Features_ROBS,
                               dd=points_EMSR171_tot4_Grading):
    points_EMSR171_feat_function, points_EMSR171_grad_function = concat_Fea_grading_EX3(bands_EX3, a, aa, b, bb, c, cc,
                                                                                        d, dd, i, train_size)
    points_feat_171_function, points_grad_171_function = shuffle(points_EMSR171_feat_function,
                                                                 points_EMSR171_grad_function, random_state=i)  # 数据打乱
    points_feat_171_DataFrame_function = pd.DataFrame(points_feat_171_function)
    points_grad_171_DataFrame_function = pd.DataFrame(points_grad_171_function)
    xbands_171_function = points_feat_171_DataFrame_function[bands_EX3].values
    yclass_171_function = points_grad_171_DataFrame_function['grading'].values
    return xbands_171_function, yclass_171_function


bands_171_importance = ['BAI', 'dNDWI', 'dNBR', 'NDVI_pre', 'dVARI', 'dNDVI', 'VARI_pre', 'NDVI_post', 'NDWI_pre',
                        'NBR_post', 'NDWI_post', 'NBR_pre', 'MSRren_post', 'CIre_pre', 'dMSRren', 'MSRren_pre',
                        'VARI_post', 'dCIre', 'NDVIre1_pre']

bands_171_irrelevant = ["BAI", "dNBR", "NDVI_pre", "dVARI", "dNDVI", "VARI_pre", "NDVI_post", "NDWI_pre", "NBR_post",
                        "MSRren_post", "dMSRren", "VARI_post", "NDVIre1_post", "dNDVIre1", "NDVIre2_pre", "dNDVIre2"]

# Classifier definition
etc_function_171 = ExtraTreesClassifier(n_estimators=701, min_samples_split=2, min_samples_leaf=1, criterion='entropy',
                                        random_state=90, n_jobs=-1)

"""
     Classification and writing confusion matrix===============>>>>Extremely randomized trees

 171 fire, 2300 samples of each class, and the features variable combination is bands_ 171_ importance_ new

"""

for j in range(1, 51):
    with open("Experiment3_EMSR171_etc_2300_importance.csv",
              "a") as csvfile_etc:
        writer_etc = csv.writer(csvfile_etc)
        name_171_1 = "Experiment3_EMSR171_etc_2300-importance"
        writer_etc.writerow([name_171_1])
        csvfile_etc.flush()
        csvfile_etc.close()

        xbands_171, yclass_171 = parameter_need_data171_EX3(i=j, train_size=2300, bands_EX3=bands_171_importance)

        sk = StratifiedKFold(n_splits=10)
        with open("Experiment3_EMSR171_etc_2300_importance.csv",
                  "a") as csvfile_etc_2:
            writer_etc_2 = csv.writer(csvfile_etc_2)
            name_171_2 = "Experiment3_EMSR171_etc_2300-importance-" + str(j) + "th"
            writer_etc_2.writerow([name_171_2])
            writer_etc_2.writerows([['Completely', 'Highly', 'Moderately', 'Negligible']])
            csvfile_etc_2.flush()
            csvfile_etc_2.close()

            # Layered cross validation
            for train_index, test_index in sk.split(xbands_171, yclass_171):
                with open("Experiment3_EMSR171_etc_2300_importance.csv",
                          "a") as csvfile_etc_3:
                    writer_etc_3 = csv.writer(csvfile_etc_3)
                    X_train, X_test = xbands_171[train_index], xbands_171[test_index]
                    y_train, y_test = yclass_171[train_index], yclass_171[test_index]

                    etc_function_171.fit(X_train, y_train)
                    etc_y_pre = etc_function_171.predict(X_test)
                    etc_confusion_matrix_fu_etc = confusion_matrix(y_test, etc_y_pre, labels=[4, 3, 2, 1])
                    # Save confusion matrix
                    writer_etc_3.writerows(etc_confusion_matrix_fu_etc)
                    csvfile_etc_3.write("\n")
                    csvfile_etc_3.flush()
                    csvfile_etc_3.close()
