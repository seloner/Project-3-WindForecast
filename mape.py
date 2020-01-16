import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    i=0
    real_i=0
    sum=0
    while i<len(y_true):
        #only if actual is not zero calculate 
        if(y_true[i]!=0):
            sum+=abs((y_true[i]-y_pred[i])/y_true[i])
            real_i+=1
        i+=1
    return (sum*100/real_i)
   
def calculate_mape(actual,prediction):
    columns_length=len(actual.columns)
    i=0
    sum=0
    while i<columns_length:
        a_column=actual[actual.columns[i]]
        p_column=prediction[:,i]
        sum+=mean_absolute_percentage_error(a_column,p_column)

        i+=1
    return sum/columns_length