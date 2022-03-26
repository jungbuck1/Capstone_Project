###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,8))

    # Constants
    bar_width = 0.3
    colors = ['#EBF10F','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, x = 0.63, y = 1.05)
    # Tune the subplot layout
    # Refer - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html for more details on the arguments
    pl.subplots_adjust(left = 0.125, right = 1.2, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.3)    
    pl.tight_layout()
    pl.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1] # original [::-1] I don't understand the double colons???
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create the plot
    fig = pl.figure(figsize = (12,9))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#AEB42A', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#0F889A', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  

# This is where it originally ended

def train_test_baseline(features, outcome):
    '''Splits data into train, test groups and creates naive predictor baseline.
    Input: features, outcome
    Output: X_train, X_test, y_train, y_test, accuracy, fscore'''
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        outcome, 
                                                        test_size = 0.2, 
                                                        random_state = 0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))
    
    TP = np.sum(outcome)
    FP = outcome.count() - TP
    TN = 0
    FN = 0
    
    # Calculate accuracy, precision and recall
    accuracy = (TP + TN)/(TP + TN + FP + FN) # np.sum(income)/income.shape[0]
    recall = TP / (TP + FN) # np.sum(income)/np.sum(income)
    precision = TP / (TP + FP) # np.sum(income)/income.shape[0]

    # Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    beta = 0.5
    fscore = (1+beta**2)*((precision*recall)/((beta**2)*precision + recall))

    # Print the results 
    print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

    return X_train, X_test, y_train, y_test, accuracy, fscore

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    Inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test), then get predictions on 
    # the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=1)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

def learning_model_comparison(classifiers, X_train, X_test, y_train, y_test, accuracy, fscore):
    '''Takes a list of learning models and returns training time, accuracy, and f-score result
    comparisons in bar graph form.
    Input: classifiers, X_train, X_test, y_train, y_test, accuracy, fscore
    Output: bar chart comparisons of training time, accuracy, f-score'''
    # Initialize the three models
    clf_A = classifiers[0]
    clf_B = classifiers[1]
    clf_C = classifiers[2]

    # Calculate the number of samples for 1%, 10%, and 100% of the training data
    samples_100 = len(y_train)
    samples_10 = int(0.1*samples_100)
    samples_1 = int(0.01*samples_100)

    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    evaluate(results, accuracy, fscore)

def top_five_features(learning_model, X_train, y_train):
    '''Takes in a learning model and creates a plot comparing the 
    top five features.
    Input: learning_model, X_train, y_train
    Output: bar chart of top five feature_importances_'''
    # Import a supervised learning model that has 'feature_importances_'
    clf = learning_model

    # Train the supervised model on the training set using .fit(X_train, y_train)
    model = clf.fit(X_train, y_train)

    # Extract the feature importances using .feature_importances_ 
    importances = model.feature_importances_

    # Plot
    feature_plot(importances, X_train, y_train);
    
def feature_processing(data):
    '''Takes a clean hotel dataframe and separates into outcome and features.
    It then normalizes numerical data and one-hot encodes categorical data
    for all the features.
    Input: clean hotel data
    Output: 1. normalized, encoded features
            2. outcome'''
    outcome = data['is_canceled']
    features_raw = data.drop('is_canceled', axis=1)
    
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                 'adults', 'children', 'babies', 'previous_cancellations', 
                 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 
                 'adr', 'required_car_parking_spaces', 'total_of_special_requests']

    features_scaled = pd.DataFrame(data = features_raw)
    features_scaled[numerical] = scaler.fit_transform(features_raw[numerical])
    
    # One-hot encode the 'features_scaled' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_scaled)

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))
    # print all the features
    print(encoded)
    
    return features_final, outcome
    
    