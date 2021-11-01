import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets


algo2 = ['Decision Tree','K Nearest Neightbour','Logistic Regession','Naïve Bayes','Random Forest','Support Vector Classification']

dataPahang2 = {'Accuracy':[0.75,0.76,0.74,0.75,0.77,0.71],
        'F1-score weighted':[0.76,0.75,0.70,0.72,0.76,0.68]}    
dfPahang2 = pd.DataFrame(dataPahang2,index=algo2)

dataKedah2 = {'Accuracy':[0.74,0.80,0.76,0.72,0.75,0.70],
        'F1-score weighted':[0.74,0.80,0.75,0.71,0.74,0.67]}    
dfKedah2 = pd.DataFrame(dataKedah2,index=algo2)

dataSelangor2 = {'Accuracy':[0.81,0.83,0.77,0.75,0.83,0.74],
        'F1-score weighted':[0.81,0.83,0.76,0.74,0.83,0.72]}    
dfSelangor2 = pd.DataFrame(dataSelangor2,index=algo2)

dataJohor2 = {'Accuracy':[0.76,0.81,0.75,0.74,0.83,0.77],
        'F1-score weighted':[0.76,0.81,0.74,0.72,0.83,0.75]}    
dfJohor2 = pd.DataFrame(dataJohor2,index=algo2)

dataPerak2 = {'Accuracy':[0.81,0.85,0.80,0.80,0.90,0.79],
        'F1-score weighted':[0.81,0.85,0.80,0.80,0.90,0.77]}    
dfPerak2 = pd.DataFrame(dataPerak2,index=algo2)

dataKelantan2 = {'Accuracy':[0.82,0.88,0.85,0.82,0.87,0.81],
        'F1-score weighted':[0.82,0.88,0.83,0.82,0.87,0.80]}    
dfKelantan2 = pd.DataFrame(dataKelantan2,index=algo2)

dataMelaka2 = {'Accuracy':[0.81,0.79,0.76,0.75,0.79,0.77],
        'F1-score weighted':[0.81,0.78,0.75,0.73,0.78,0.75]}    
dfMelaka2 = pd.DataFrame(dataMelaka2,index=algo2)

dataNegeri_Sembilan2 = {'Accuracy':[0.73,0.71,0.73,0.67,0.77,0.73],
        'F1-score weighted':[0.73,0.71,0.69,0.66,0.77,0.70]}    
dfNegeri_Sembilan2 = pd.DataFrame(dataNegeri_Sembilan2,index=algo2)

dataPerlis2 = {'Accuracy':[0.69,0.73,0.74,0.72,0.78,0.74],
        'F1-score weighted':[0.69,0.71,0.73,0.74,0.77,0.73]}    
dfPerlis2 = pd.DataFrame(dataPerlis2,index=algo2)

dataPulau_Pinang2 = {'Accuracy':[0.75,0.82,0.75,0.78,0.83,0.72],
        'F1-score weighted':[0.74,0.82,0.73,0.77,0.82,0.70]}    
dfPulau_Pinang2 = pd.DataFrame(dataPulau_Pinang2,index=algo2)

dataSarawak2 = {'Accuracy':[0.70,0.77,0.71,0.70,0.74,0.70],
        'F1-score weighted':[0.70,0.77,0.70,0.69,0.74,0.69]}    
dfSarawak2 = pd.DataFrame(dataSarawak2,index=algo2)

dataSabah2 = {'Accuracy':[0.71,0.72,0.66,0.63,0.75,0.66],
        'F1-score weighted':[0.71,0.72,0.64,0.62,0.76,0.65]}    
dfSabah2 = pd.DataFrame(dataSabah2,index=algo2)

dataTerengganu2 = {'Accuracy':[0.77,0.81,0.79,0.75,0.80,0.77],
        'F1-score weighted':[0.77,0.79,0.77,0.73,0.78,0.75]}    
dfTerengganu2 = pd.DataFrame(dataTerengganu2,index=algo2)

dataKL2 = {'Accuracy':[0.78,0.85,0.80,0.74,0.83,0.74],
        'F1-score weighted':[0.78,0.84,0.79,0.73,0.83,0.72]}    
dfKL2 = pd.DataFrame(dataKL2,index=algo2)

dataLabuan2 = {'Accuracy':[0.66,0.69,0.66,0.63,0.69,0.63],
        'F1-score weighted':[0.66,0.67,0.64,0.65,0.68,0.62]}    
dfLabuan2 = pd.DataFrame(dataLabuan2,index=algo2)

dataPutrajaya2 = {'Accuracy':[0.67,0.72,0.68,0.65,0.66,0.74],
        'F1-score weighted':[0.67,0.72,0.66,0.64,0.65,0.69]}    
dfPutrajaya2 = pd.DataFrame(dataPutrajaya2,index=algo2)

#####################################################################
algo3 = ['Decision Tree','Linear regression','Ridge Regression','Support vector']
dataPahang4 = {'R2':[0.78,-7.4,-0.22,-0.59],
        'MAE':[0.25,1.44,0.77,0.78]}
dfPahang4 = pd.DataFrame(dataPahang4,index=algo3)

dataKedah4 = {'R2':[0.77,0.76,0.77,0.73],
        'MAE':[0.25,0.45,0.45,0.43]}
dfKedah4 = pd.DataFrame(dataKedah4,index=algo3)

dataSelangor4 = {'R2':[0.89,-3.08,0.65,0.95],
        'MAE':[301.84,1657.32,557.17,207.29]}
dfSelangor4 = pd.DataFrame(dataSelangor4,index=algo3)

dataJohor4 = {'R2':[0.86,-39.36,0.65,0.91],
        'MAE':[79.78,1108.33,146.79,74.30]}
dfJohor4 = pd.DataFrame(dataJohor4,index=algo3)

dataKelantan4 = {'R2':[0.93,0.68,0.81,0.94],
        'MAE':[42.98,92.51,70.88,43.41]}
dfKelantan4 = pd.DataFrame(dataKelantan4,index=algo3)

dataPerak4 = {'R2':[0.88,0.55,0.82,0.92],
        'MAE':[50.72,104.28,71.03,40.42]}
dfPerak4 = pd.DataFrame(dataPerak4,index=algo3)

dataMelaka4 = {'R2':[0.72,-12.52,-0.17,0.81],
        'MAE':[42.39,261.75,89.45,34.72]}
dfMelaka4 = pd.DataFrame(dataMelaka4,index=algo3)

dataNegeri_Sembilan4 = {'R2':[0.56,0.69,0.69,0.82],
        'MAE':[82.88,74.90,73.62,48.21]}
dfNegeri_Sembilan4 = pd.DataFrame(dataNegeri_Sembilan4,index=algo3)

dataPulau_Pinang4 = {'R2':[0.89,0.64,0.76,0.92],
        'MAE':[57.43,99.44,77.25,50.80]}
dfPulau_Pinang4 = pd.DataFrame(dataPulau_Pinang4,index=algo3)

dataPerlis4 = {'R2':[0.68,-8.84,0.24,0.55],
        'MAE':[3.83,23.62,7.45,5.3]}
dfPerlis4 = pd.DataFrame(dataPerlis4,index=algo3)

dataSarawak4 = {'R2':[0.86,0.88,0.88,0.88],
        'MAE':[107.79,123.44,120.14,104.70]}
dfSarawak4 = pd.DataFrame(dataSarawak4,index=algo3)

dataSabah4 = {'R2':[0.90,0.91,0.91,0.89],
        'MAE':[74.81,119.25,118.98,99.25]}
dfSabah4 = pd.DataFrame(dataSabah4,index=algo3)
dataTerengganu4 = {'R2':[0.85,-5.66,0.70,0.86],
        'MAE':[34.09,195.51,48.02,33.87]}
dfTerengganu4 = pd.DataFrame(dataTerengganu4,index=algo3)

dataKL4 = {'R2':[0.70,0.86,0.86,0.88],
        'MAE':[117.88,95.55,94.42,79.22]}
dfKL4 = pd.DataFrame(dataKL4,index=algo3)
dataLabuan4 = {'R2':[0.61,-4096.99,-1.65,0.48],
        'MAE':[10.63,705.32,31.22,14.57]}
dfLabuan4 = pd.DataFrame(dataLabuan4,index=algo3)
dataPutrajaya4 = {'R2':[0.65,0.61-5.68,0.09,0.48],
        'MAE':[4.90,16.67,7.70,6.61]}
dfPutrajaya4 = pd.DataFrame(dataPutrajaya4,index=algo3)



def app():
    boruta_rank_cases = pd.read_csv('boruCases.csv')
    boruta_rank_admi = pd.read_csv('boruAdmi.csv')
    rfe_cases = pd.read_csv('rfeCases.csv')

    st.markdown('### Feature Selection & Predictive Modelling')
    st.write(" In this section, we will focus on utilizing machine learning to predict daily cases of all states in Malaysia and to predict people admitted to hospital daily.")
    st.markdown('#### 1.) Predict daily cases daily cases of all states in malaysia')
    st.write("")
    st.markdown('##### Feature Selection')
    st.write("""In this section we focus on extracting meaningful independant variables from the dataset provided by Ministry Of Health by using two algorithm which is BORUTA and RFE. Since we are using two algorithm to do feature selection, we will use the independant variables that has rank below 30 from each algorithm.
    """)
    with st.expander("Strong Features Indicated by Boruta of Each States"):
        a1,a2,a3,a4 = st.columns((1,1,1,1))

        a1.write("Johor")
        a2.write("Selangor")
        a3.write("Sarawak")
        a4.write("Sabah")

        a1.dataframe(boruta_rank_cases[['Feature0','Ranking0']].sort_values(by="Ranking0").reset_index().drop('index',axis='columns').rename(columns={'Feature0': 'Feature', 'Ranking0': 'Ranking'}))
        a2.dataframe(boruta_rank_cases[['Feature1','Ranking1']].sort_values(by="Ranking1").reset_index().drop('index',axis='columns').rename(columns={'Feature1': 'Feature', 'Ranking1': 'Ranking'}))
        a3.dataframe(boruta_rank_cases[['Feature2','Ranking2']].sort_values(by="Ranking2").reset_index().drop('index',axis='columns').rename(columns={'Feature2': 'Feature', 'Ranking2': 'Ranking'}))
        a4.dataframe(boruta_rank_cases[['Feature3','Ranking3']].sort_values(by="Ranking3").reset_index().drop('index',axis='columns').rename(columns={'Feature3': 'Feature', 'Ranking3': 'Ranking'}))
        
        a1.write("Putrajaya")
        a2.write("Perlis")
        a3.write("Pahang")
        a4.write("Pulau Pinang")

        a1.dataframe(boruta_rank_cases[['Feature4','Ranking4']].sort_values(by="Ranking4").reset_index().drop('index',axis='columns').rename(columns={'Feature4': 'Feature', 'Ranking4': 'Ranking'}))
        a2.dataframe(boruta_rank_cases[['Feature5','Ranking5']].sort_values(by="Ranking5").reset_index().drop('index',axis='columns').rename(columns={'Feature5': 'Feature', 'Ranking5': 'Ranking'}))
        a3.dataframe(boruta_rank_cases[['Feature6','Ranking6']].sort_values(by="Ranking6").reset_index().drop('index',axis='columns').rename(columns={'Feature6': 'Feature', 'Ranking6': 'Ranking'}))
        a4.dataframe(boruta_rank_cases[['Feature7','Ranking7']].sort_values(by="Ranking7").reset_index().drop('index',axis='columns').rename(columns={'Feature7': 'Feature', 'Ranking7': 'Ranking'}))
        
        a1.write("Perak")
        a2.write("Labuan")
        a3.write("Negeri Sembilan")
        a4.write("Melaka")

        a1.dataframe(boruta_rank_cases[['Feature8','Ranking8']].sort_values(by="Ranking8").reset_index().drop('index',axis='columns').rename(columns={'Feature8': 'Feature', 'Ranking8': 'Ranking'}))
        a2.dataframe(boruta_rank_cases[['Feature9','Ranking9']].sort_values(by="Ranking9").reset_index().drop('index',axis='columns').rename(columns={'Feature9': 'Feature', 'Ranking9': 'Ranking'}))
        a3.dataframe(boruta_rank_cases[['Feature10','Ranking10']].sort_values(by="Ranking10").reset_index().drop('index',axis='columns').rename(columns={'Feature10': 'Feature', 'Ranking10': 'Ranking'}))
        a4.dataframe(boruta_rank_cases[['Feature11','Ranking11']].sort_values(by="Ranking11").reset_index().drop('index',axis='columns').rename(columns={'Feature11': 'Feature', 'Ranking11': 'Ranking'}))
        
        a1.write("Kuala Lumpur")
        a2.write("Kedah")
        a3.write("Kelantan")
        a4.write("Terengganu")

        a1.dataframe(boruta_rank_cases[['Feature12','Ranking12']].sort_values(by="Ranking12").reset_index().drop('index',axis='columns').rename(columns={'Feature12': 'Feature', 'Ranking12': 'Ranking'}))
        a2.dataframe(boruta_rank_cases[['Feature13','Ranking13']].sort_values(by="Ranking13").reset_index().drop('index',axis='columns').rename(columns={'Feature13': 'Feature', 'Ranking13': 'Ranking'}))
        a3.dataframe(boruta_rank_cases[['Feature14','Ranking14']].sort_values(by="Ranking14").reset_index().drop('index',axis='columns').rename(columns={'Feature14': 'Feature', 'Ranking14': 'Ranking'}))
        a4.dataframe(boruta_rank_cases[['Feature15','Ranking15']].sort_values(by="Ranking15").reset_index().drop('index',axis='columns').rename(columns={'Feature15': 'Feature', 'Ranking15': 'Ranking'}))

    with st.expander("Strong Features Indicated by RFE of Each States"):
        b1,b2,b3,b4 = st.columns((1,1,1,1))

        b1.write("Johor")
        b2.write("Selangor")
        b3.write("Sarawak")
        b4.write("Sabah")

        b1.dataframe(rfe_cases[['Feature0','Ranking0']].sort_values(by="Ranking0").reset_index().drop('index',axis='columns').rename(columns={'Feature0': 'Feature', 'Ranking0': 'Ranking'}))
        b2.dataframe(rfe_cases[['Feature1','Ranking1']].sort_values(by="Ranking1").reset_index().drop('index',axis='columns').rename(columns={'Feature1': 'Feature', 'Ranking1': 'Ranking'}))
        b3.dataframe(rfe_cases[['Feature2','Ranking2']].sort_values(by="Ranking2").reset_index().drop('index',axis='columns').rename(columns={'Feature2': 'Feature', 'Ranking2': 'Ranking'}))
        b4.dataframe(rfe_cases[['Feature3','Ranking3']].sort_values(by="Ranking3").reset_index().drop('index',axis='columns').rename(columns={'Feature3': 'Feature', 'Ranking3': 'Ranking'}))
        
        b1.write("Putrajaya")
        b2.write("Perlis")
        b3.write("Pahang")
        b4.write("Pulau Pinang")

        b1.dataframe(rfe_cases[['Feature4','Ranking4']].sort_values(by="Ranking4").reset_index().drop('index',axis='columns').rename(columns={'Feature4': 'Feature', 'Ranking4': 'Ranking'}))
        b2.dataframe(rfe_cases[['Feature5','Ranking5']].sort_values(by="Ranking5").reset_index().drop('index',axis='columns').rename(columns={'Feature5': 'Feature', 'Ranking5': 'Ranking'}))
        b3.dataframe(rfe_cases[['Feature6','Ranking6']].sort_values(by="Ranking6").reset_index().drop('index',axis='columns').rename(columns={'Feature6': 'Feature', 'Ranking6': 'Ranking'}))
        b4.dataframe(rfe_cases[['Feature7','Ranking7']].sort_values(by="Ranking7").reset_index().drop('index',axis='columns').rename(columns={'Feature7': 'Feature', 'Ranking7': 'Ranking'}))
        
        b1.write("Perak")
        b2.write("Labuan")
        b3.write("Negeri Sembilan")
        b4.write("Melaka")

        b1.dataframe(rfe_cases[['Feature8','Ranking8']].sort_values(by="Ranking8").reset_index().drop('index',axis='columns').rename(columns={'Feature8': 'Feature', 'Ranking8': 'Ranking'}))
        b2.dataframe(rfe_cases[['Feature9','Ranking9']].sort_values(by="Ranking9").reset_index().drop('index',axis='columns').rename(columns={'Feature9': 'Feature', 'Ranking9': 'Ranking'}))
        b3.dataframe(rfe_cases[['Feature10','Ranking10']].sort_values(by="Ranking10").reset_index().drop('index',axis='columns').rename(columns={'Feature10': 'Feature', 'Ranking10': 'Ranking'}))
        b4.dataframe(rfe_cases[['Feature11','Ranking11']].sort_values(by="Ranking11").reset_index().drop('index',axis='columns').rename(columns={'Feature11': 'Feature', 'Ranking11': 'Ranking'}))
        
        b1.write("Kuala Lumpur")
        b2.write("Kedah")
        b3.write("Kelantan")
        b4.write("Terengganu")

        b1.dataframe(rfe_cases[['Feature12','Ranking12']].sort_values(by="Ranking12").reset_index().drop('index',axis='columns').rename(columns={'Feature12': 'Feature', 'Ranking12': 'Ranking'}))
        b2.dataframe(rfe_cases[['Feature13','Ranking13']].sort_values(by="Ranking13").reset_index().drop('index',axis='columns').rename(columns={'Feature13': 'Feature', 'Ranking13': 'Ranking'}))
        b3.dataframe(rfe_cases[['Feature14','Ranking14']].sort_values(by="Ranking14").reset_index().drop('index',axis='columns').rename(columns={'Feature14': 'Feature', 'Ranking14': 'Ranking'}))
        b4.dataframe(rfe_cases[['Feature15','Ranking15']].sort_values(by="Ranking15").reset_index().drop('index',axis='columns').rename(columns={'Feature15': 'Feature', 'Ranking15': 'Ranking'}))
    st.write("")
    st.markdown("##### Models' Performance in Predicting Daily Cases of All States in Malaysia")
    st.markdown("__Classification__")

    with st.expander("See Classification Models' Performance of Each States"):
        do1,do2 =st.columns((2,1))
        do2.write("")

        do1.write("Pahang")
        do1.table(dfPahang2)
        do2.write("")

        do1.write("Kedah")
        do1.table(dfKedah2)
        do2.write("")

        do1.write("Selangor")
        do1.table(dfSelangor2)
        do2.write("")

        do1.write("Johor")
        do1.table(dfJohor2)
        do2.write("")

        do1.write("Perak")
        do1.table(dfPerak2)
        do2.write("")

        do1.write("Kelantan")
        do1.table(dfKelantan2)
        do2.write("")

        do1.write("Melaka")
        do1.table(dfMelaka2)
        do2.write("")

        do1.write("Negeri Sembilan")
        do1.table(dfNegeri_Sembilan2)
        do2.write("")

        do1.write("Sabah")
        do1.table(dfSabah2)
        do2.write("")

        do1.write("Terengganu")
        do1.table(dfTerengganu2)
        do2.write("")

        do1.write("Sarawak")
        do1.table(dfSarawak2)
        do2.write("")

        do1.write("Perlis")
        do1.table(dfPerlis2)
        do2.write("")

        do1.write("Kuala Lumpur")
        do1.table(dfKL2)
        do2.write("")

        do1.write("Labuan")
        do1.table(dfLabuan2)
        do2.write("")

        do1.write("Pahang")
        do1.table(dfPahang2)
        do2.write("")

        do1.write("Putrajaya")
        do1.table(dfPutrajaya2)
        do2.write("")

    st.markdown("__Regression__")
    with st.expander("See Regression Models' Performance of Each States"):
        co1,co2 =st.columns((2,1))
        co2.write("")

        co1.write("Pahang")
        co1.table(dfPahang4)
        co2.write("")

        co1.write("Kedah")
        co1.table(dfKedah4)
        co2.write("")

        co1.write("Selangor")
        co1.table(dfSelangor4)
        co2.write("")

        co1.write("Johor")
        co1.table(dfJohor4)
        co2.write("")

        co1.write("Perak")
        co1.table(dfPerak4)
        co2.write("")

        co1.write("Kelantan")
        co1.table(dfKelantan4)
        co2.write("")

        co1.write("Melaka")
        co1.table(dfMelaka4)
        co2.write("")

        co1.write("Negeri Sembilan")
        co1.table(dfNegeri_Sembilan4)
        co2.write("")

        co1.write("Sabah")
        co1.table(dfSabah4)
        co2.write("")

        co1.write("Terengganu")
        co1.table(dfTerengganu4)
        co2.write("")

        co1.write("Sarawak")
        co1.table(dfSarawak4)
        co2.write("")

        co1.write("Perlis")
        co1.table(dfPerlis4)
        co2.write("")

        co1.write("Kuala Lumpur")
        co1.table(dfKL4)
        co2.write("")

        co1.write("Labuan")
        co1.table(dfLabuan4)
        co2.write("")

        co1.write("Pahang")
        co1.table(dfPahang4)
        co2.write("")

        co1.write("Putrajaya")
        co1.table(dfPutrajaya4)
        co2.write("")

    st.markdown('#### Conclusion ')
    st.markdown(""" ###### We found that for the regression problem, MLP performed the best as it was the best model for 11 states out of 16 states. The next best performing model is support vector regression which is best performed for 3 out of 16  states. Lastly is decision tree regression and ridge regression which was the best model for 1 state respectively. From this we can conclude that for this data set MLP is the best model to use.""")
    st.markdown(""" ###### For classification approach, random forest classification was the majority best performing model as it is the best for 8 states out of 16. Following it is k nearest neighbor which was best performing for 5 states out of 16. Lastly SVC(RBF) performed best in 2 states and decision tree classification performed best in 1 state. From this we can conclude that for this data set random forest is the best model to use. """)
    st.write(" ")
    st.markdown('### 2.) Predict people admitted to hospital daily')
    st.markdown('##### Feature Selection')
    st.write("""" 
        For this question we separate into 3 cases, without feature selection, with top 100 features and with top 50 features.For feature selection, we did not do RFE because it takes too long to load. Hence to compensate that, we we separate into 3 cases. For the case without feature selection, MLP was the majority 
        best performing model within 9 states and following it is support vector regression (linear) with 7 states. However, we only show the best case in this website which is with top 100 features. The remaining cases are well written in the documentation.
    """)
    with st.expander("Strong Features Indicated by Boruta of Each States"):
        e1,e2,e3,e4 = st.columns((1,1,1,1))

        e1.write("Johor")
        e2.write("Selangor")
        e3.write("Sarawak")
        e4.write("Sabah")

        e1.dataframe(boruta_rank_admi[['Feature0','Ranking0']].sort_values(by="Ranking0").reset_index().drop('index',axis='columns').rename(columns={'Feature0': 'Feature', 'Ranking0': 'Ranking'}))
        e2.dataframe(boruta_rank_admi[['Feature1','Ranking1']].sort_values(by="Ranking1").reset_index().drop('index',axis='columns').rename(columns={'Feature1': 'Feature', 'Ranking1': 'Ranking'}))
        e3.dataframe(boruta_rank_admi[['Feature2','Ranking2']].sort_values(by="Ranking2").reset_index().drop('index',axis='columns').rename(columns={'Feature2': 'Feature', 'Ranking2': 'Ranking'}))
        e4.dataframe(boruta_rank_admi[['Feature3','Ranking3']].sort_values(by="Ranking3").reset_index().drop('index',axis='columns').rename(columns={'Feature3': 'Feature', 'Ranking3': 'Ranking'}))
        
        e1.write("Putrajaya")
        e2.write("Perlis")
        e3.write("Pahang")
        e4.write("Pulau Pinang")

        e1.dataframe(boruta_rank_admi[['Feature4','Ranking4']].sort_values(by="Ranking4").reset_index().drop('index',axis='columns').rename(columns={'Feature4': 'Feature', 'Ranking4': 'Ranking'}))
        e2.dataframe(boruta_rank_admi[['Feature5','Ranking5']].sort_values(by="Ranking5").reset_index().drop('index',axis='columns').rename(columns={'Feature5': 'Feature', 'Ranking5': 'Ranking'}))
        e3.dataframe(boruta_rank_admi[['Feature6','Ranking6']].sort_values(by="Ranking6").reset_index().drop('index',axis='columns').rename(columns={'Feature6': 'Feature', 'Ranking6': 'Ranking'}))
        e4.dataframe(boruta_rank_admi[['Feature7','Ranking7']].sort_values(by="Ranking7").reset_index().drop('index',axis='columns').rename(columns={'Feature7': 'Feature', 'Ranking7': 'Ranking'}))
        
        e1.write("Perak")
        e2.write("Labuan")
        e3.write("Negeri Sembilan")
        e4.write("Melaka")

        e1.dataframe(boruta_rank_admi[['Feature8','Ranking8']].sort_values(by="Ranking8").reset_index().drop('index',axis='columns').rename(columns={'Feature8': 'Feature', 'Ranking8': 'Ranking'}))
        e2.dataframe(boruta_rank_admi[['Feature9','Ranking9']].sort_values(by="Ranking9").reset_index().drop('index',axis='columns').rename(columns={'Feature9': 'Feature', 'Ranking9': 'Ranking'}))
        e3.dataframe(boruta_rank_admi[['Feature10','Ranking10']].sort_values(by="Ranking10").reset_index().drop('index',axis='columns').rename(columns={'Feature10': 'Feature', 'Ranking10': 'Ranking'}))
        e4.dataframe(boruta_rank_admi[['Feature11','Ranking11']].sort_values(by="Ranking11").reset_index().drop('index',axis='columns').rename(columns={'Feature11': 'Feature', 'Ranking11': 'Ranking'}))
        
        e1.write("Kuala Lumpur")
        e2.write("Kedah")
        e3.write("Kelantan")
        e4.write("Terengganu")

        e1.dataframe(boruta_rank_admi[['Feature12','Ranking12']].sort_values(by="Ranking12").reset_index().drop('index',axis='columns').rename(columns={'Feature12': 'Feature', 'Ranking12': 'Ranking'}))
        e2.dataframe(boruta_rank_admi[['Feature13','Ranking13']].sort_values(by="Ranking13").reset_index().drop('index',axis='columns').rename(columns={'Feature13': 'Feature', 'Ranking13': 'Ranking'}))
        e3.dataframe(boruta_rank_admi[['Feature14','Ranking14']].sort_values(by="Ranking14").reset_index().drop('index',axis='columns').rename(columns={'Feature14': 'Feature', 'Ranking14': 'Ranking'}))
        e4.dataframe(boruta_rank_admi[['Feature15','Ranking15']].sort_values(by="Ranking15").reset_index().drop('index',axis='columns').rename(columns={'Feature15': 'Feature', 'Ranking15': 'Ranking'}))

    st.markdown("#### Model's Performance in Predicting People Admitted to Hospital daily")
    st.markdown("__Regression__")
    with st.expander("See Regression Models' Performance of Each States"):
        algo3 = ['Decision Tree','Linear regression','Ridge Regression','Support vector']
        dataPahang3 = {'R2':[0.86,0.93,0.93,0.95],
                'MAE':[42.33,59.57,58.08,48.62]}
        dfPahang3 = pd.DataFrame(dataPahang3,index=algo3)

        dataKedah3 = {'R2':[0.94,0.93,0.94,0.97],
                'MAE':[59.76,76.82,69.37,43.00]}
        dfKedah3 = pd.DataFrame(dataKedah3,index=algo3)

        dataSelangor3 = {'R2':[0.92,0.93,0.93,0.94],
                'MAE':[321.84,325.87,317.92,247.83]}
        dfSelangor3 = pd.DataFrame(dataSelangor3,index=algo3)

        dataJohor3 = {'R2':[0.81,0.93,0.94,0.94],
                'MAE':[112.29,87.24,82.70,69.30]}
        dfJohor3 = pd.DataFrame(dataJohor3,index=algo3)

        dataKelantan3 = {'R2':[0.94,0.92,0.93,0.96],
                'MAE':[51.68,67.11,63.64,41.13]}
        dfKelantan3 = pd.DataFrame(dataKelantan3,index=algo3)

        dataPerak3 = {'R2':[0.90,0.84,0.86,0.94],
                'MAE':[62.52,85.01,77.10,42.44]}
        dfPerak3 = pd.DataFrame(dataPerak3,index=algo3)

        dataMelaka3 = {'R2':[0.70,0.61,0.66,0.81],
                'MAE':[96.82,57.80,54.26,38.01]}
        dfMelaka3 = pd.DataFrame(dataMelaka3,index=algo3)

        dataNegeri_Sembilan3 = {'R2':[0.51,0.64,0.66,0.71],
                'MAE':[94.54,90.03,86.49,68.98]}
        dfNegeri_Sembilan3 = pd.DataFrame(dataNegeri_Sembilan3,index=algo3)

        dataPulau_Pinang3 = {'R2':[0.65,0.56,0.60,0.71],
                'MAE':[5.06,6.4,5.79,4.64]}
        dfPulau_Pinang3 = pd.DataFrame(dataPulau_Pinang3,index=algo3)

        dataPerlis3 = {'R2':[0.65,0.56,0.60,0.71],
                'MAE':[5.06,6.4,5.79,4.64]}
        dfPerlis3 = pd.DataFrame(dataPerlis3,index=algo3)

        dataSarawak3 = {'R2':[0.91,0.90,0.91,0.90],
                'MAE':[99.79,124.44,119.14,114.70]}
        dfSarawak3 = pd.DataFrame(dataSarawak3,index=algo3)
        dataSabah3 = {'R2':[0.86,0.96,0.96,0.96],
                'MAE':[99.81,67.25,65.98,114.25]}
        dfSabah3 = pd.DataFrame(dataSabah3,index=algo3)
        dataTerengganu3 = {'R2':[0.93,0.89,0.90,0.94],
                'MAE':[30.89,41.51,37.02,25.87]}
        dfTerengganu3 = pd.DataFrame(dataTerengganu3,index=algo3)

        dataKL3 = {'R2':[0.89,0.82,0.83,0.91],
            'MAE':[108.88,131.55,127.42,98.22]}
        dfKL3 = pd.DataFrame(dataKL3,index=algo3)

        dataLabuan3 = {'R2':[0.84,0.48,0.55,0.84],
                'MAE':[8.63,18.32,17.22,9.57]}
        dfLabuan3 = pd.DataFrame(dataLabuan3,index=algo3)

        dataPutrajaya3 = {'R2':[0.65,0.61,0.67,0.89],
                'MAE':[4.53,6.17,5.67,3.13]}
        dfPutrajaya3 = pd.DataFrame(dataPutrajaya3,index=algo3)

        bo1,bo2 =st.columns((2,1))
        bo2.write("")

        bo1.write("Pahang")
        bo1.table(dfPahang3)
        bo2.write("")

        bo1.write("Kedah")
        bo1.table(dfKedah3)
        bo2.write("")

        bo1.write("Selangor")
        bo1.table(dfSelangor3)
        bo2.write("")

        bo1.write("Johor")
        bo1.table(dfJohor3)
        bo2.write("")

        bo1.write("Perak")
        bo1.table(dfPerak3)
        bo2.write("")

        bo1.write("Kelantan")
        bo1.table(dfKelantan3)
        bo2.write("")

        bo1.write("Melaka")
        bo1.table(dfMelaka3)
        bo2.write("")

        bo1.write("Negeri Sembilan")
        bo1.table(dfNegeri_Sembilan3)
        bo2.write("")

        bo1.write("Sabah")
        bo1.table(dfSabah3)
        bo2.write("")

        bo1.write("Terengganu")
        bo1.table(dfTerengganu3)
        bo2.write("")

        bo1.write("Sarawak")
        bo1.table(dfSarawak3)
        bo2.write("")

        bo1.write("Perlis")
        bo1.table(dfPerlis3)
        bo2.write("")

        bo1.write("Kuala Lumpur")
        bo1.table(dfKL3)
        bo2.write("")

        bo1.write("Labuan")
        bo1.table(dfLabuan3)
        bo2.write("")

        bo1.write("Pahang")
        bo1.table(dfPahang3)
        bo2.write("")

        bo1.write("Putrajaya")
        bo1.table(dfPutrajaya3)
        bo2.write("")

    st.markdown('#### Conclusion ')
    st.markdown(""" ###### To evaluate the model we use r square, mean absolute error, mean square error, and root mean squared error. R2 measures the goodness of fit of the model. MAE measures the magnitude of difference between the prediction of an observation and the true value. For MAE, top 100 features increased slightly by 0.2 and for top 50 features increased more by 6. For MSE, the top 100 features increased by approximately 2k and for the top 50 it increased by approximately 4k. For RMSE, the top 100 features increased slightly by 0.6 and top 50 features decreased by approximately 4. Hence to sum up, the model with top 100 features selected by boruta performed the best in predicting number of peopl admitted to hospital daily """)
