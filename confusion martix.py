from sklearn import metrics
import pandas as pd

pred_list = [0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,1,1,0]
actual_list = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

actual_list = pd.Series(actual_list, name='Actual')
pred_list = pd.Series(pred_list, name='Predicted')

print(pd.crosstab(actual_list, pred_list))

#print accuracy of model
print('Accuracy Score: ', metrics.accuracy_score(actual_list, pred_list))

#print precision value of model
print('Precision Score: ', metrics.precision_score(actual_list, pred_list))

#print recall value of model
print('Recall Score: ',metrics.recall_score(actual_list, pred_list))
