import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split


class Regression: 
    dataset: pd.DataFrame
    target: str
    __X: pd.DataFrame 
    __y: pd.DataFrame
    __X_train: pd.DataFrame
    __X_test: pd.DataFrame
    __y_train: pd.DataFrame
    __y_test: pd.DataFrame
    __comp_table: dict
    
    def __init__(self, dataset, target_col) -> None: 
        self.dataset = dataset 
        self.target = target_col
        self.__X, self.__y = self.dataset.drop([self.target], axis=1), self.dataset[self.target]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size=0.25, random_state=0)
        return 
    
    def get_coeff(self): 
        X, y = self.__X_train, self.__y_train
        X["constant"] = np.ones((len(X), 1))
        X, y = X.values, y.values 
       
        beta = np.linalg.pinv(X).dot(y)
        return beta 
    
    def get_predictions(self): 
        X = self.__X_test 
        X["constant"] = np.ones((len(X), 1))
        X = X.values 
        beta = self.get_coeff()
        yprime = X.dot(beta)
        return yprime 
    
    def construct_error_table(self):
        yprime = self.get_predictions()
        self.__comp_table ={
            "TestValues": self.__y_test, 
            "Predictions": yprime, 
            "residue": self.__y_test - yprime, 
            "error": (self.__y_test - yprime)**2
        }
        return pd.DataFrame(self.__comp_table)
    
    def __repr__(self) -> str:
        self.construct_error_table()
        return str(self.__comp_table) + "\n" + str(self.err()) 
    
    def err(self):
        table = self.construct_error_table()
        det_coeff = 1 - (np.sum(table["error"]))/np.sum((self.__y_test - np.mean(self.__y_test))**2)
        e = ((np.sum(table["error"]))/(len(self.__y_test) - 2))**0.5 
        errors = {
            "score": det_coeff, 
            "Se": e
        }
        return errors 
    
    def score(self): 
        return self.err()["score"]
        
         
    
    