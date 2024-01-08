import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import make_scorer
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor




# def custom_loss_func(ground_truth, predictions):
#     return np.sum(paired_distances(ground_truth, predictions))/len(ground_truth)


def main():
    df = pd.read_csv('development.csv', sep=',')

    # print(df.dtypes)
    # print(df.isna().any(axis=0)) # no null values
    # positions = df[['x', 'y']]
    # x_values = positions['x'].unique()
    # y_values = positions['y'].unique()
    # fig, ax = plt.subplots(figsize = (8,8))
    # ax.scatter(x_values, y_values)
    # plt.show()
    # print(len(x_values))
    # print(len(y_values))
    
    pmax = [f'pmax[{i}]' for i in range(18)]
    means = df[pmax].mean(axis=0)
    # print(means)
    for i in range(6):
        means.drop(means.index[means.argmax()], inplace=True)
    valid_pmax = list(means.index)
    # print(valid_pmax)

    negpmax = [f'negpmax[{i}]' for i in range(18)]
    means = df[negpmax].mean(axis=0)
    # print(means)
    for i in range(6):
        means.drop(means.index[means.argmin()], inplace=True)
    valid_negpmax = list(means.index)
    # print(valid_negpmax)

    # print(list(zip(valid_pmax, valid_negpmax))) # in pmax 15 is missing, in negpmax 3 is missing

    area = [f'area[{i}]' for i in range(18)]
    means = df[area].mean(axis=0)
    # print(means)
    for i in range(6):
        means.drop(means.index[means.argmax()], inplace=True)
    valid_area = list(means.index)
    # print(valid_area) # same indices as pmax

    tmax = [f'tmax[{i}]' for i in range(18)]
    means = df[tmax].mean(axis=0)
    # print(means)  # the readings 5, 10, 13, 15, 16, 17 occur with the same delay, which seems suspect as pads are in different positions 
                    # in the sensor, so we expect that they detect the positive peak at different times
    
    rms = [f'rms[{i}]' for i in range(18)]
    means = df[rms].mean(axis=0)
    # print(means) # apart from readings 16 and 17, there are not noticeable differences, maybe because of the impulsive nature of the noise
                    # as seen with the large values of pmax on the suspect readings
    
    noise_pmax = []
    noise_negpmax = []
    noise_area = []
    noise_tmax = []
    noise_rms = []
    for i in [5, 10, 13, 15, 16, 17]:
        noise_pmax.append(f'pmax[{i}]')
        noise_negpmax.append(f'negpmax[{i}]')
        noise_area.append(f'area[{i}]')
        noise_tmax.append(f'tmax[{i}]')
        noise_rms.append(f'rms[{i}]')

    noise = noise_pmax + noise_negpmax + noise_area + noise_tmax + noise_rms
    # print(noise)
    # print(df.shape)
    df.drop(columns=noise, inplace=True)
    # print(df.shape)

    # X = df.drop(columns=['x', 'y']).values
    # y = df[['x', 'y']].values

    # param_grid = {
    # "n_neighbors": [2, 3, 4, 5],
    # "weights": ["uniform", "distance"],
    # "algorithm": ["auto", "ball_tree", "kd_tree"],
    # "n_jobs": [-1]
    # }

    # custom_scorer = make_scorer(custom_loss_func, greater_is_better=False)
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    # gs = GridSearchCV(KNeighborsRegressor(), param_grid, scoring=custom_scorer, n_jobs=-1, cv=5)
    # gs.fit(X, y)
    # print(gs.best_score_)
    # print(gs.best_params_)

    # param_grid = {
    # "n_estimators": [100, 250, 500],
    # "criterion": ["squared_error"],
    # "max_features": ["auto", "sqrt", "log2"],
    # "random_state": [42], 
    # "n_jobs": [-1]
    # }
    
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_norm = sc.transform(X_train)
    # X_valid_norm = sc.transform(X_valid)
    # reg = KNeighborsRegressor()             
    # reg.fit(X_train , y_train)
    # y_pred = reg.predict(X_valid)
    


    df_eval = pd.read_csv('evaluation.csv', sep=',')
    df_eval.drop(columns=noise, inplace=True)



    X_train = df.drop(columns=['x', 'y']).values
    y_train = df[['x', 'y']].values
    X_test = df_eval.drop(columns='Id').values
    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_norm = sc.transform(X_train)
    # X_test_norm = sc.transform(X_test)         
    # reg.fit(X_train_norm , y_train)               # try min_samples_split=5
    
    reg = RandomForestRegressor(n_estimators = 30, criterion= 'squared_error', max_features= 'sqrt',  
                                 random_state=42)   # score = 9.102
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print('finito')

    header = ['Id', 'Predicted']
    with open("submission.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in df_eval['Id']:
            writer.writerow([i, ''.join((str(y_pred[i, 0]), '|', str(y_pred[i, 1])))])

  

    
if __name__ == "__main__":
    main()