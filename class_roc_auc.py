import matplotlib.pyplot as plt
from itertools import cycle
from numpy import interp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from nilearn.image import index_img
from nilearn.decoding import Decoder
import nibabel as nib

from scipy import interp

LABEL_1 = 'AD'
LABEL_2 = 'Normal'

PATH_TO_REP = 'data/'
metadata = pd.read_csv(PATH_TO_REP + 'metadata.csv')
testdata = pd.read_csv(PATH_TO_REP + 'test.csv')

smc_mask = ((metadata.Label == LABEL_1) | (metadata.Label == LABEL_2)).values.astype('bool')
y_train = (metadata[smc_mask].Label == LABEL_1).astype(np.int32).values
smc_mask = ((testdata.Label == LABEL_1) | (testdata.Label == LABEL_2)).values.astype('bool')
y_test = (testdata[smc_mask].Label == LABEL_1).astype(np.int32).values

X_train_img = nib.funcs.concat_images(metadata['Path'])
X_test_img = nib.funcs.concat_images(testdata['Path'])

decoder = Decoder(estimator='svc', standardize=True, n_jobs=20, screening_percentile=10)

fpr = dict()
tpr = dict()
roc_auc = dict()
accuracy=[]

cv = StratifiedKFold(n_splits=5)
fold = 0

for train_index, val_index in cv.split(metadata['Path'],metadata['Label']):
      #X_train_img = nib.funcs.concat_images(X_train.values[train_index])
      #X_val_img = nib.funcs.concat_images(X_train.values[val_index])
      decoder.fit(index_img(X_train_img, train_index), y_train[train_index])
      prediction = decoder.predict(index_img(X_train_img, val_index))
      predValues = decoder.decision_function(index_img(X_train_img, val_index))
      accuracy.append((prediction == y_train[val_index]).sum() / float(len(val_index)))
      fpr[fold], tpr[fold], _ = roc_curve(y_train[val_index], predValues)
      roc_auc[fold] = auc(fpr[fold], tpr[fold])
      fold += 1

## Plot ROC curve
cv_num = 5
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(cv_num)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(cv_num):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= cv_num
fpr["mean"] = all_fpr
tpr["mean"] = mean_tpr
roc_auc["mean"] = auc(fpr["mean"], tpr["mean"])
plt.figure()
lw = 2
plt.plot(fpr["mean"], tpr["mean"],
         label='Mean ROC curve (AUC = {0:0.2f})'
               ''.format(roc_auc["mean"]),
         color='deeppink', linestyle=':', linewidth=4)

colors = cycle(["#A079BF", "#006FA6", "#B79762", "#FF4A46", "#008941"])
for i, color in zip(range(cv_num), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC fold {0} (AUC = {1:0.4f})'
             ''.format(i + 1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic 5-fold cross validation')
plt.legend(loc="lower right")
plt.savefig("plot_dir/10_per_auc_result.png")


prediction = decoder.predict(X_test_img)
#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(prediction,y_test)
print('Confusion Matrix : \n', cm1)
# Confusion Matrix :
#  [[31  2]
#  [ 4 23]]

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)

##plotting
from nilearn.plotting import plot_stat_map
AD_weight_img = decoder.coef_img_[1] ##AD
template_bg_img="/home/share/dataShare/template/resized_template110_110_110.nii.gz"
cut_slides=(-56,-41,-23,-4,4,21,41,49,56,72)
plot_stat_map(AD_weight_img, bg_img=template_bg_img,
  title='SVM weights', colorbar=True, output_file='plot_dir/weight_img_z_selected_template_bg_img.png', display_mode='z', cut_coords=cut_slides)
