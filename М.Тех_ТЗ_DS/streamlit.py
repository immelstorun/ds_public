import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import streamlit.components.v1 as components

#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report