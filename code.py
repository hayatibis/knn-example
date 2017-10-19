import scipy.io as si
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

NY_F = si.loadmat('Ny_Features.mat', squeeze_me=True, struct_as_record=False)
NY_T = si.loadmat('Ny_Train.mat', squeeze_me=True, struct_as_record=False)
R_F = si.loadmat('Rome_Features.mat', squeeze_me=True, struct_as_record=False)
R_T = si.loadmat('Rome_Train.mat', squeeze_me=True, struct_as_record=False)

NY_Features = NY_F['feats']
NY_Train = NY_T['imInfo']
Rome_Features = R_F['feats']
Rome_Train = R_T['imInfo']

k = 11

Vector = []
X_Matrix = []
GPS_target = []
Labels = []

for x in NY_Features:
    for y in NY_Train:
        if y.name == x.name:
            Vector.append(np.concatenate([[x.name], [1], x.tiny16, x.gist, [y.longitude], [y.latitude], ['NY_Label']]))
            break

for x in Rome_Features:
    for y in Rome_Train:
        if y.name == x.name:
            Vector.append(
                np.concatenate([[x.name], [1], x.tiny16, x.gist, [y.longitude], [y.latitude], ['Rome_Label']]))
            break

Vector = np.array(Vector)
np.random.shuffle(Vector)

for x in range(len(Vector)):
    X_Matrix.append(Vector[x][1:1282])

X_Matrix = np.array(X_Matrix).astype('float')

for x in range(len(Vector)):
    GPS_target.append(Vector[x][1282:1284])

GPS_target = np.array(GPS_target).astype('float')


#Functions
def accuracyCalc(test_labels, predictedLabels):
    counter = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictedLabels[i]:
            counter += 1

    acc = 100 * ((counter) / (len(test_labels)))
    return acc
    
def closed_form(X_Matrix, GPS_target):
    x = X_Matrix
    x_T = X_Matrix.transpose()

    y = GPS_target

    mult = np.dot(x_T, x)

    inv = np.linalg.pinv(mult)
    temp = np.dot(inv, x_T)
    w = np.dot(temp, y)

    return w

def predict(closed_w, test_features):
    est_coordinates = np.dot(test_features, closed_w)
    return est_coordinates
    
def kNN(train_gps, train_labels, test_gps, k):
    mNN = KNeighborsClassifier(n_neighbors=k)
    mNN.fit(train_gps, train_labels)

    predLabels = mNN.predict(test_gps)
    return predLabels
#End of functions

test_features = X_Matrix[42774:]
train_features = X_Matrix[:42774]

test_gps = GPS_target[42774:]
train_gps = GPS_target[:42774]

#closed form solution theta calculation
closed_w = closed_form(train_features, train_gps)

est_coor = predict(closed_w, test_features)

for x in range(len(Vector)):
    Labels.append(Vector[x][1284])

Labels = np.array(Labels)

test_labels = Labels[42774:]
train_labels = Labels[:42774]

predictedLabels = kNN(train_gps, train_labels, est_coor, k)

print("Accuracy:")
print(accuracyCalc(test_labels, predictedLabels))



