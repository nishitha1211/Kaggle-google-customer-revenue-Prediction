from math import isnan

import pandas as pd
import numpy as np
import json
import gc
from sklearn import model_selection,neural_network,metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

path = "C:/Users/achan/PycharmProjects/KaggleProject-MachineLearning/train1.csv"

testPath = "C:/Users/achan/PycharmProjects/KaggleProject-MachineLearning/test2.csv"

def load_data_to_dataframe(csv_path=path, nrows=None):
    columns_json = ['device', 'geoNetwork', 'totals', 'trafficSource']

    dframe = pd.read_csv(csv_path,
                     converters={column: json.loads for column in columns_json},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)

    for column in columns_json:
        column_df = json_normalize(dframe[column])
        column_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_df.columns]
        dframe = dframe.drop(column, axis=1).merge(column_df, right_index=True, left_index=True)
    # print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return dframe

c = load_data_to_dataframe(path)


print(c.describe())

attributeMap = dict()
count = 0.0
def normalizeColumn(data, attributes):
    global count
    for i in sorted(data[attributes].unique(), reverse=True):
        if i not in attributeMap:
            attributeMap[i] = count
            count = count + 1.0
    data[attributes] = data[attributes].map(attributeMap)
    return data

# fullVisitorId and sessionId are object type

columnList = ['date', 'fullVisitorId', 'sessionId', 'visitId', 'visitNumber', 'visitStartTime']

for column in c:
    if column not in columnList:
        c[column] = c[column].astype('str')

for column in c:
    if column not in columnList:
        normalizeColumn(c, column)


#  find columns with constant values
constant_columns = []
for col in c.columns:
    if len(c[col].value_counts()) == 1:
        constant_columns.append(col)

# remove columns with constant values
for column in constant_columns:
    del c[column]

# print(c.describe())

test = load_data_to_dataframe(testPath)
# print(test.columns)
for column in test:
    if column not in columnList:
        test[column] = test[column].astype('str')

for column in test:
    if column not in columnList:
        normalizeColumn(test, column)

for column in constant_columns:
    del test[column]

y=c['totals.transactionRevenue'].values



#dividing the data for training and validation
X_train,X_validation,y_train,y_validation=model_selection.train_test_split(c, y, test_size=0.20, random_state=0)
result_val = pd.DataFrame({"fullVisitorId":X_validation['fullVisitorId'], 'transactionRevenue': X_validation["totals.transactionRevenue"]})
del X_train['totals.transactionRevenue']
del X_validation["totals.transactionRevenue"]
rev = dict(map(reversed, attributeMap.items()))



# plot for  geoNetwork Continent

geoContinent = dict()
for i in c['geoNetwork.continent']:
    if i not in geoContinent.keys():
        geoContinent[i] = 1
    else:
        geoContinent[i] += 1

x = list(geoContinent.keys())
y = []
xLabel = []
for i in x:
    for key, val in attributeMap.items():
        if i == val:
            xLabel.append(key)
            y.append(geoContinent.get(i))

plt.pie(y, labels=xLabel)
plt.show()

#plot for device browsers

deviceBrowserColumns = dict()
for i in c['device.browser']:
    if i not in deviceBrowserColumns.keys():
        deviceBrowserColumns[i] = 1
    else:
        deviceBrowserColumns[i] += 1

x = deviceBrowserColumns.keys()
columnNames = []
reversed_dictionary = dict(map(reversed, attributeMap.items()))
for i in x:
    columnNames.append((reversed_dictionary.get(i)))

y = deviceBrowserColumns.values()
plt.plot(y,columnNames)
plt.show()

# plot for ChannelGrouping

channelGrouping = dict()
for i in c['channelGrouping']:
    if i not in channelGrouping.keys():
        channelGrouping[i] = 1
    else:
        channelGrouping[i] += 1

x = list(channelGrouping.keys())
y = []
xLabel = []
for i in x:
    for key, val in attributeMap.items():
        if i == val:
            xLabel.append(key)
            y.append(channelGrouping.get(i))

plt.bar(xLabel, y, alpha=0.5)
plt.setp(plt.xticks()[1], rotation=30, ha='right')
plt.show()

#plot for operatingsystem

deviceOsColumns = dict()
for i in c['device.operatingSystem']:
    if i not in deviceOsColumns.keys():
        deviceOsColumns[i] = 1
    else:
        deviceOsColumns[i] += 1
x = deviceOsColumns.keys()
columnNames = []
reversed_dictionary = dict(map(reversed, attributeMap.items()))
for i in x:
    columnNames.append((reversed_dictionary.get(i)))
y = deviceOsColumns.values()
plt.plot(y, columnNames, alpha=1)
plt.show()


# #Neural network
# print("Neural Network")
# #build the neural network model and fit the model
# neural_net = neural_network.MLPClassifier(hidden_layer_sizes=(5,),activation="relu",alpha=0.0001)
# neural_net.fit(X_train,y_train)
#
# #predict the model
# y_pred = neural_net.predict(X_validation)
# result_val['predictedRevenue'] = y_pred
#
# z = []
# for i in result_val['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z.append(0.0)
#     else:
#         z.append(float(temp))
# result_val['predictedRevenue'] = z
# del z
# n = []
# for i in result_val['transactionRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         n.append(0.0)
#     else:
#         n.append(float(temp))
# result_val['transactionRevenue'] = n
# del n
#
# total = result_val['transactionRevenue'].sum()
# result_val['total']=result_val['transactionRevenue']
# result_val['total'] = total
#
# #RME value
# print('RMSE value')
# print(np.sqrt(metrics.mean_squared_error(np.log1p(result_val['predictedRevenue']).values, np.log1p(result_val['transactionRevenue']) )))
#
# #classification Report
# print('classification report')
# print(classification_report(y_validation, y_pred))
#
# #accuracy of the model
# print('accuracy')
# print(metrics.accuracy_score(y_validation, y_pred))
#
# #predict the test and save it to the csv file
# y_test_pred = neural_net.predict(test)
# result = pd.DataFrame({"fullVisitorId":test['fullVisitorId']})
# result['predictedRevenue'] = y_test_pred
#
# z2= []
# for i in result['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z2.append(0.0)
#     else:
#         z2.append(float(temp))
# result['predictedRevenue'] = z2
# del z2
#
# result = result.groupby('fullVisitorId')['predictedRevenue'].sum().reset_index()
# result.columns = ["fullVisitorId", "predictedLogRevenue"]
# result["predictedLogRevenue"] = np.log1p(result["predictedLogRevenue"])
# min = result["predictedLogRevenue"].min()
# result["predictedLogRevenue"] -= min
# result.to_csv('output.csv',index=True)
# print(result.head(6))
# print('\n')
# del result
# del total
# del neural_net
# gc.collect()

# #SVM
# print("SVM")
# #build the neural network model and fit the model
# svm=SVC(kernel= 'sigmoid', gamma= 1e-1,C= 10,degree=2)
# svm.fit(X_train,y_train)
#
# #predict the model
# y_pred = svm.predict(X_validation)
# result_val['predictedRevenue'] = y_pred

# z = []
# for i in result_val['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z.append(0.0)
#     else:
#         z.append(float(temp))
# result_val['predictedRevenue'] = z
# del z
# n = []
# for i in result_val['transactionRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         n.append(0.0)
#     else:
#         n.append(float(temp))
# result_val['transactionRevenue'] = n
# del n

# total = result_val['transactionRevenue'].sum()
# result_val['total']=result_val['transactionRevenue']
# result_val['total'] = total
#
# #RME value
# print('RMSE value')
# print(np.sqrt(metrics.mean_squared_error(np.log1p(result_val['predictedRevenue']).values, np.log1p(result_val['transactionRevenue']) )))
#
# #classification Report
# print('classification report')
# print(classification_report(y_validation, y_pred))
#
# #accuracy of the model
# print('accuracy')
# print(metrics.accuracy_score(y_validation, y_pred))
#
# #predict the test and save it to the csv file
# y_test_pred = svm.predict(test)
# result = pd.DataFrame({"fullVisitorId":test['fullVisitorId']})
# # result['predictedRevenue'] = y_test_pred
#
# z2= []
# for i in result['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z2.append(0.0)
#     else:
#         z2.append(float(temp))
# result['predictedRevenue'] = z2
# del z2

# result = result.groupby('fullVisitorId')['predictedRevenue'].sum().reset_index()
# result.columns = ["fullVisitorId", "predictedLogRevenue"]
# result["predictedLogRevenue"] = np.log1p(result["predictedLogRevenue"])
# min = result["predictedLogRevenue"].min()
# result["predictedLogRevenue"] -= min
# result.to_csv('output1.csv', index=True)
# print(result.head(6))
# print('\n')


#Random Forest
print("Random Forest")
#build the neural network model and fit the model
random_forest=RandomForestClassifier(n_estimators=10,criterion= 'gini', max_depth= 3, max_features= 'sqrt')

random_forest.fit(X_train,y_train)

#predict the model
y_pred = random_forest.predict(X_validation)
result_val['predictedRevenue'] = y_pred

z = []
for i in result_val['predictedRevenue']:
    temp=rev.get(i)
    if isnan(float(temp)):
        z.append(0.0)
    else:
        z.append(float(temp))
result_val['predictedRevenue'] = z
del z
n = []
for i in result_val['transactionRevenue']:
    temp=rev.get(i)
    if isnan(float(temp)):
        n.append(0.0)
    else:
        n.append(float(temp))
result_val['transactionRevenue'] = n
del n

total = result_val['transactionRevenue'].sum()
result_val['total']=result_val['transactionRevenue']
result_val['total'] = total

#RME value
print('RMSE value')
print(np.sqrt(metrics.mean_squared_error(np.log1p(result_val['predictedRevenue']).values, np.log1p(result_val['transactionRevenue']) )))

#classification Report
print('classification report')
print(classification_report(y_validation, y_pred))

#accuracy of the model
print('accuracy')
print(metrics.accuracy_score(y_validation, y_pred))

#predict the test and save it to the csv file
y_test_pred = random_forest.predict(test)
result = pd.DataFrame({"fullVisitorId":test['fullVisitorId']})
result['predictedRevenue'] = y_test_pred

z2= []
for i in result['predictedRevenue']:
    temp=rev.get(i)
    if isnan(float(temp)):
        z2.append(0.0)
    else:
        z2.append(float(temp))
result['predictedRevenue'] = z2
del z2

result = result.groupby('fullVisitorId')['predictedRevenue'].sum().reset_index()
result.columns = ["fullVisitorId", "predictedLogRevenue"]
result["predictedLogRevenue"] = np.log1p(result["predictedLogRevenue"])
min = result["predictedLogRevenue"].min()
result["predictedLogRevenue"] -= min
result.to_csv('output1.csv', index=True)
print(result.head(6))
print('\n')


# #Adaboosting
# print("Adaboosting")
# #build the neural network model and fit the model
# adaboost=AdaBoostClassifier(n_estimators = 100, learning_rate= 0.5, algorithm='SAMME.R' ,random_state=1)
#
# adaboost.fit(X_train,y_train)
#
# #predict the model
# y_pred = adaboost.predict(X_validation)
# result_val['predictedRevenue'] = y_pred

# z = []
# for i in result_val['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z.append(0.0)
#     else:
#         z.append(float(temp))
# result_val['predictedRevenue'] = z
# del z
# n = []
# for i in result_val['transactionRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         n.append(0.0)
#     else:
#         n.append(float(temp))
# result_val['transactionRevenue'] = n
# del n

# total = result_val['transactionRevenue'].sum()
# result_val['total']=result_val['transactionRevenue']
# result_val['total'] = total
#
# #RME value
# print('RMSE value')
# print(np.sqrt(metrics.mean_squared_error(np.log1p(result_val['predictedRevenue']).values, np.log1p(result_val['transactionRevenue']) )))
#
# #classification Report
# print('classification report')
# print(classification_report(y_validation, y_pred))
#
# #accuracy of the model
# print('accuracy')
# print(metrics.accuracy_score(y_validation, y_pred))
#
# #predict the test and save it to the csv file
# y_test_pred = adaboost.predict(test)
# result = pd.DataFrame({"fullVisitorId":test['fullVisitorId']})
# result['predictedRevenue'] = y_test_pred

# z2= []
# for i in result['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z2.append(0.0)
#     else:
#         z2.append(float(temp))
# result['predictedRevenue'] = z2
# del z2

# result = result.groupby('fullVisitorId')['predictedRevenue'].sum().reset_index()
# result.columns = ["fullVisitorId", "predictedLogRevenue"]
# result["predictedLogRevenue"] = np.log1p(result["predictedLogRevenue"])
# min = result["predictedLogRevenue"].min()
# result["predictedLogRevenue"] -= min
# result.to_csv('output1.csv', index=True)
# print(result.head(6))
# print('\n')
#
#
# #Bagging
#
# print("Bagging")
# #build the neural network model and fit the model
# bg=BaggingClassifier(n_estimators=10, bootstrap=False, warm_start=False)
# bg.fit(X_train,y_train)
#
# #predict the model
# y_pred = bg.predict(X_validation)
# result_val['predictedRevenue'] = y_pred
#
# z = []
# for i in result_val['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z.append(0.0)
#     else:
#         z.append(float(temp))
# result_val['predictedRevenue'] = z
# del z
# n = []
# for i in result_val['transactionRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         n.append(0.0)
#     else:
#         n.append(float(temp))
# result_val['transactionRevenue'] = n
# del n
#
# total = result_val['transactionRevenue'].sum()
# result_val['total']=result_val['transactionRevenue']
# result_val['total'] = total
# #RME value
# print('RMSE value')
# print(np.sqrt(metrics.mean_squared_error(np.log1p(result_val['predictedRevenue']).values, np.log1p(result_val['transactionRevenue']) )))
#
# #classification Report
# print('classification report')
# print(classification_report(y_validation, y_pred))
#
# #accuracy of the model
# print('accuracy')
# print(metrics.accuracy_score(y_validation, y_pred))
#
# #predict the test and save it to the csv file
# y_test_pred = bg.predict(test)
# result = pd.DataFrame({"fullVisitorId":test['fullVisitorId']})
# result['predictedRevenue'] = y_test_pred
# z2= []
# for i in result['predictedRevenue']:
#     temp=rev.get(i)
#     if isnan(float(temp)):
#         z2.append(0.0)
#     else:
#         z2.append(float(temp))
# result['predictedRevenue'] = z2
# del z2
# result = result.groupby('fullVisitorId')['predictedRevenue'].sum(). reset_index()
# result.columns = ["fullVisitorId", "predictedLogRevenue"]
# result["predictedLogRevenue"] = np.log1p(result["predictedLogRevenue"])
# min = result["predictedLogRevenue"].min()
# result["predictedLogRevenue"] -= min
# result.to_csv('output1.csv', index=True)
# print(result.head(6))
# print('\n')


