import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(60, len(df_scaled)):
        X.append(df_scaled[i-60:i])
        y.append(df_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    return train_test_split(X, y, test_size=0.2, shuffle=False)
