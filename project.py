import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._scorer import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


def custom_loss_func(ground_truth, predictions):
    return np.sum(paired_distances(ground_truth, predictions))/len(ground_truth)


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
    
    # fare analisi distribuzione features


    pmax = [f'pmax[{i}]' for i in range(18)] # fare plot delle statistiche
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

    area = [f'area[{i}]' for i in range(18)] # maybe remove area
    means = df[area].mean(axis=0)
    # print(means)
    for i in range(6):
        means.drop(means.index[means.argmax()], inplace=True)
    valid_area = list(means.index)
    # print(valid_area) # same indices as pmax

    tmax = [f'tmax[{i}]' for i in range(18)] # maybe remove tmax
    means = df[tmax].mean(axis=0)
    std = df[tmax].std(axis=0)
    # print(means)  # the readings 5, 10, 13, 15, 16, 17 occur with the same delay, which seems suspect as pads are in different positions 
    # print(std)                # in the sensor, so we expect that they detect the positive peak at different times
    
    rms = [f'rms[{i}]' for i in range(18)] # maybe remove rms
    means = df[rms].mean(axis=0)
    # print(means) # apart from readings 16 and 17, there are not noticeable differences, maybe because of the impulsive nature of the noise
                    # as seen with the large values of pmax on the suspect readings
    
    # try to plot graphs to show outliers as noise
    
    noise_pmax = []
    noise_negpmax = []
    noise_area = []
    noise_tmax = []
    noise_rms = []
    for i in [0, 7, 12, 16, 17]:   #15
        noise_negpmax.append(f'negpmax[{i}]')
        # noise_area.append(f'area[{i}]')

    for i in [0, 7, 12, 16, 17]: 
        noise_pmax.append(f'pmax[{i}]')
        noise_area.append(f'area[{i}]')
        
    
    for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15]: 
        df = df[df[f'negpmax[{i}]'] < 0]
        df[f'pkpk[{i}]'] = df[f'pmax[{i}]'] + np.abs(df[f'negpmax[{i}]'])  # np.abs
   
    for i in range(18): 
        noise_rms.append(f'rms[{i}]')
        noise_tmax.append(f'tmax[{i}]')

        

    # print(noise_rms)

    noise = noise_pmax + noise_negpmax + noise_area + noise_tmax + noise_rms
    df.drop(columns=noise, inplace=True)
    # print(noise)
    # print(df.shape)
    
    # print(df.head)

    df = df.sample(frac=1).reset_index(drop=True)

    # print(df.head)




    # print(df.shape)
    
    # print(y.shape)

    # plt.figure(figsize=(14, 8))
    # df.boxplot(column=['negpmax[15]','negpmax[14]','negpmax[13]','negpmax[11]','negpmax[10]','negpmax[9]','negpmax[8]','negpmax[6]','negpmax[5]','negpmax[4]','negpmax[3]','negpmax[2]', 'negpmax[1]'], vert=False,fontsize=14)
    # plt.title('BoxPlot negpmax')
    # plt.xlabel('Time, ns',fontsize=16)
    # plt.ylabel('negpmax',fontsize=16)
    # plt.show()

   

    custom_scorer = make_scorer(custom_loss_func, greater_is_better=False)
    param_grid = {'n_estimators': [50, 100, 200],
              'criterion': ['squared_error'],
              'max_features': ['sqrt', 'log2'],
              'random_state': [42],
              'n_jobs': [-1]      
    }

    # param_grid = {'n_neighbors': [5, 7, 9],
    #           'weights': ['uniform', 'distance'],
    #           'algorithm': ['auto']    
    # }

    X = df.drop(columns=['x', 'y']).values
    y = df[['x', 'y']].values
    X_train_valid = X
    y_train_valid = y
    


    
    # gs = GridSearchCV(RandomForestRegressor(), param_grid, scoring=custom_scorer, n_jobs=-1, cv=5)
    # gs.fit(X_train_valid, y_train_valid)
    # print(gs.best_score_)   # -4.077039667366269
    # print(gs.best_estimator_)  # {'criterion': 'squared_error', 'max_features': 'sqrt', 'n_estimators': 200, 'n_jobs': -1, 'random_state': 42}
    # print(gs.best_params_)  # {'algorithm': 'auto', 'n_neighbors': 9, 'weights': 'distance'}

    
    


    
    # df = df[valids]
    # print(df.shape)
    


    df_eval = pd.read_csv('evaluation.csv', sep=',')
   
    for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15]:  
        df_eval[f'pkpk[{i}]'] = df_eval[f'pmax[{i}]'] + np.abs(df_eval[f'negpmax[{i}]'])

    df_eval.drop(columns=noise, inplace=True)

    

    # print(df.shape)
    # print(df_eval.shape)

    X_train = df.drop(columns=['x', 'y']).values
    y_train = df[['x', 'y']].values
    X_test = df_eval.drop(columns='Id').values
    
    # reg = KNeighborsRegressor(n_neighbors=9, weights='distance', algorithm='auto')  #standard scaler su neighbors
    # reg = LinearRegression()
    reg = RandomForestRegressor(n_estimators = 200, criterion='squared_error', max_features='sqrt', random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)           
    y_pred = reg.predict(X_test)        # with pkpk and pmax 15 n=50 score=4.828, n=200 score=4.675
                              
    
    print('finito')
    feature_names = df.drop(columns=['x', 'y']).columns.values
    for tuple in sorted(zip(feature_names, reg.feature_importances_),  key=lambda x: x[1],reverse=True):
        print(tuple)
  

    header = ['Id', 'Predicted']
    with open("submission.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in df_eval['Id']: 
            writer.writerow([i, ''.join((str(round(y_pred[i, 0], 1)), '|', str(round(y_pred[i, 1], 1))))])
  

    
if __name__ == "__main__":
    main()