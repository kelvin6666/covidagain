import streamlit as st
import __future__
import numpy as np
import pandas as pd
from sklearn import datasets
import os
import numpy as np
import datetime as dt
import plotly.express as px
import calendar
from functools import reduce
from datetime import timedelta
from dateutil.relativedelta import *
from sklearn import preprocessing
from scipy import sign
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff

def importCsv(name, listName): #function to import every csv in a folder
        dir_name = os.getcwd()
        for file in os.listdir(dir_name + '\\'  +name):
            df = pd.read_csv(dir_name + '\\'  + name +'\\' +file, error_bad_lines=False)
            df.name = file
            listName.append(df)

def app():
    ########## Python ############
    move = pd.read_csv("mysejahtera/checkin_malaysia.csv")
    pop = pd.read_csv('population.csv')
    epidemic_list = [] #declare list to store epidemic data
    importCsv('epidemic data',epidemic_list )
    vaxcination_list = [] #declare list to store vaccination dataset
    importCsv('vaxcination data',vaxcination_list )

    mysejahtera = [] #Declare list to store mysejahtera dataset
    importCsv('Mysejahtera',mysejahtera)

    vax_msia = vaxcination_list[0]
    vax_state = vaxcination_list[1]
    vax_msia['date'] = pd.to_datetime(vax_msia['date'],errors='coerce') #change object to datetime
    cases_msia = epidemic_list[0]
    cases_state = epidemic_list[1]
    cases_msia['date'] = pd.to_datetime(cases_msia['date'],errors='coerce')
    vax_cases = vax_msia.merge(cases_msia,how='inner',on='date')
    vax_cases['date'] = pd.to_datetime(vax_cases['date'],errors='coerce')
    ########## Python ############
    st.write(" ")
    st.markdown('### What clustering algorithm is suitable to find different clusters from the dataset? How do you quantify the risk?')
    st.write("For this question we will be mainly using three datasets, checkin_malaysia.csv (movement data), tests_malaysia.csv and cases_malaysia.csv to find different clusters and quantify the risk. We will also compare the performance between DBScan, KMean and Agglomerative Clustering")
    st.markdown("#### With movement data ")
    with st.expander("KMean"):
        g1,g2,g3 =  st.columns((0.5,2,0.5))
        X=move.drop("date",axis=1)
        scaler = MinMaxScaler()
        scaler.fit(X)
        X=scaler.transform(X)
        inertia = []
        for i in range(1,11):
            kmeans = KMeans(
                n_clusters=i, init="k-means++",
                n_init=10,
                tol=1e-04, random_state=42
            )
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        fig_move_kmean = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
        fig_move_kmean.update_layout(title="Inertia vs Cluster Number (To find out optimal number of cluster)",xaxis=dict(range=[0,11],title="Cluster Number"),
                        yaxis={'title':'Inertia'},
                        annotations=[
                dict(
                    x=3,
                    y=inertia[2],
                    xref="x",
                    yref="y",
                    text="Elbow",
                    showarrow=True,
                    arrowhead=7,
                    ax=20,
                    ay=-40
                )
            ])

        kmeansPolar = KMeans(
            n_clusters=3, init="k-means++",
            n_init=10,
            tol=1e-04, random_state=42
        )
        kmeansPolar.fit(X)
        clusters= pd.DataFrame(X,columns=move.drop("date",axis=1).columns)
        clusters['label']=kmeansPolar.labels_
        polar=clusters.groupby("label").mean().reset_index()
        polar=pd.melt(polar,id_vars=["label"])
        fig_polar_kmean = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=500,width=800)
        
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        y_km = kmeans.fit_predict(X)
        move['cluster'] = y_km
        

        PLOTmove_kmean = go.Figure()

        for C in list(move.cluster.unique()):
            
            PLOTmove_kmean.add_trace(go.Scatter3d(x = move[move.cluster == C]['checkins'],
                                        y = move[move.cluster == C]['unique_ind'],
                                        z = move[move.cluster == C]['unique_loc'],
                                        mode = 'markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(C)))
            

        PLOTmove_kmean.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                        scene = dict(xaxis=dict(title = 'checkins', titlefont_color = 'white'),
                                        yaxis=dict(title = 'unique_ind', titlefont_color = 'white'),
                                        zaxis=dict(title = 'unique_loc', titlefont_color = 'white')),
                        font = dict(family = "Gilroy", color  = 'white', size = 12))

        g2.plotly_chart(PLOTmove_kmean,use_container_width=True)
        g2.plotly_chart(fig_move_kmean,use_container_width=True)
        st.write("From the 3d plot above we can see that we set the number of clusters to be 3 as the elbow indicates that the optimal number of cluster should be 3. To analyze the meaning of each cluster, we will use a polar graph to help us to understand the characteristic of each cluster")
        h1,h2 = st.columns((1,1))
        h1.plotly_chart(fig_polar_kmean,use_container_width=True)
        h2.write("""
        From the polar plot, we can see that for to be classified as cluster 0, the number of check ins,unique location and unique individual of that day must be very high. This is because the main way for Covid-19 to spread is pread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. A person can be infected when aerosols or droplets containing the virus are inhaled or come directly into contact with the eyes, nose, or mouth.
        With high number of check ins，unique location and unique individual, it shows that there are many places are crowded with different individuals. This scenario is perfect for an outbreak of Covid-19 to happen. Hence, we would like to say that cluster 0 brings the meaning of "very risky to cause widespread of covid-19".

        For cluster 2, it exhibits a similar properties as cluster 0. However it has a lower number of check ins. This shows that on that day, they are a moderate number of check ins at many places around malaysia with many different individuals. This condition also has the posibility to cause the widespread of covid-19, but it has fewer check ins meaning that 
        the number of people going out on that day is fewer. Hence, we would like to conclude that cluster 2 has the meaning of "risky to cause widespread of covid-19"

        For cluster 1, we can see that it has the least check ins, unique individual and unique location in a day. This shows that on that day, there are very few people heading out. This is an ideal setup
        to stop the widespread of Covid-19. Hence, we would like to say that cluster 1 has the meaning of "less risky to cause widespread of covid-19"
        """)

    with st.expander("DBScan"):
        h1,h2,h3 =  st.columns((0.5,2,0.5))
        dbscan = DBSCAN(eps=0.3,min_samples = 10)
        dbscan.fit(X)
        y_db = dbscan.labels_
        move['cluster'] = y_db

        PLOTdbscan = go.Figure()

        for C in list(move.cluster.unique()):
            
            PLOTdbscan.add_trace(go.Scatter3d(x = move[move.cluster == C]['checkins'],
                                        y = move[move.cluster == C]['unique_ind'],
                                        z = move[move.cluster == C]['unique_loc'],
                                        mode = 'markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(C)))
            

        PLOTdbscan.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                        scene = dict(xaxis=dict(title = 'checkins', titlefont_color = 'white'),
                                        yaxis=dict(title = 'unique_ind', titlefont_color = 'white'),
                                        zaxis=dict(title = 'unique_loc', titlefont_color = 'white')),
                        font = dict(family = "Gilroy", color  = 'white', size = 12))
        
        h2.plotly_chart(PLOTdbscan,use_container_width=True)
        st.write("From the 3d plot above we can see that the number of cluster identified by DBScan is 2 and only one of them is from Cluster -1 and the remaining are cluster 0. We are unable to find out meaning from the cluster as there are only two clusters being identified and most of them belongs to a cluster. This probably due to the nature of the data and the algorithm of dbscan. The data is very near to each other hence DBscan will cluster all of them together. Hence DBScan is not suitable in this dataset.")
    with st.expander("Agglomerative Clustering"):
        i1,i2,i3 =  st.columns((0.5,2,0.5))
        dendo_move = ff.create_dendrogram(X)
        dendo_move.add_hline(y=1, line_width=3, line_dash="dash", line_color="green")
        dendo_move.update_layout(width=800, height=500)
        agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        agg.fit(X)
        agg_labels = agg.labels_

        move['cluster'] = agg_labels

        PLOT_agg_move = go.Figure()

        for C in list(move.cluster.unique()):
            
            PLOT_agg_move.add_trace(go.Scatter3d(x = move[move.cluster == C]['checkins'],
                                        y = move[move.cluster == C]['unique_ind'],
                                        z = move[move.cluster == C]['unique_loc'],
                                        mode = 'markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(C)))
            

        PLOT_agg_move.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                        scene = dict(xaxis=dict(title = 'checkins', titlefont_color = 'white'),
                                        yaxis=dict(title = 'unique_ind', titlefont_color = 'white'),
                                        zaxis=dict(title = 'unique_loc', titlefont_color = 'white')),
                        font = dict(family = "Gilroy", color  = 'white', size = 12))
        i2.plotly_chart(PLOT_agg_move,use_container_width=True)
        i2.plotly_chart(dendo_move,use_container_width=True)
        st.write("""From the 3d plot above we can see that we set the number of clusters to be 3 as set the threshold of its dendogram to 1 and the horizontal line intersect the graph 3 times.
        To analyze the meaning of each cluster, we will use a polar graph to help us to understand the characteristic of 
        each cluster""")
        print(X)
        print(move)
        clusters_agg_move= pd.DataFrame(X,columns= move.drop(["date","cluster"],axis=1).columns)
        clusters_agg_move['label']=agg.labels_
        polar_agg_move=clusters.groupby("label").mean().reset_index()
        polar_agg_move=pd.melt(polar_agg_move,id_vars=["label"])
        fig_polar_agg_move = px.line_polar(polar_agg_move, r="value", theta="variable", color="label", line_close=True,height=500,width=800)
        j1,j2 = st.columns((1,1))
        j1.plotly_chart(fig_polar_agg_move,use_container_width=True)
        j2.write("""
            The polar plot of Agglomerative Clustering is identical to the polar plot of Kmean. Hence, our analysis will remain the same.

            From the polar plot, we can see that for to be classified as cluster 0, the number of check ins,unique location and unique individual of that day must be very high. This is because the main way for Covid-19 to spread is spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. A person can be infected when aerosols or droplets containing the virus are inhaled or come directly into contact with the eyes, nose, or mouth.
            With high number of check ins，unique location and unique individual, it shows that there are many places are crowded with different individuals. This scenario is perfect for an outbreak of Covid-19 to happen. Hence, we would like to say that cluster 0 brings the meaning of "very risky to cause widespread of covid-19".

            For cluster 2, it exhibits a similar properties as cluster 0. However it has a lower number of check ins. This shows that on that day, they are a moderate number of check ins at many places around malaysia with many different individuals. This condition also has the posibility to cause the widespread of covid-19, but it has fewer check ins meaning that 
            the number of people going out on that day is fewer. Hence, we would like to conclude that cluster 2 has the meaning of "risky to cause widespread of covid-19"

            For cluster 1, we can see that it has the least check ins, unique individual and unique location in a day. This shows that on that day, they are very less people heading out. This is an ideal setup
            to stop the widespread of Covid-19. Hence, we would like to say that cluster 1 has the meaning of "less risky to cause widespread of covid-19"
            """)
    
    st.markdown('#### Conclusion for Clustering of Movement Data')
    st.markdown(""" ###### Both KMean and Agglomerative Clustering perform very well in identifying the cluster from the movement dataset. Both of them are able to quantify the risk of widespread of Covid-19 by using Check Ins, Unique Location and Unique Individual. """)
    st.write(" ")
    
        
    
    st.markdown("#### With Testing and Cases Data ")
    test_msia = epidemic_list[-2]
    test_msia['date'] =  pd.to_datetime(test_msia['date'],errors='coerce')
    test_msia = test_msia.merge(cases_msia,how='left',on = 'date')
    test_cluster = test_msia.iloc[:, 0:4]
    test_cluster = test_cluster.dropna()
    test_cluster["total_test"] = test_cluster['rtk-ag'] + test_cluster['pcr']
    with st.expander("KMean"):
        j1,j2,j3 =  st.columns((0.5,2,0.5))

        X=test_cluster.drop(["date"],axis=1)
        scaler = MinMaxScaler()
        scaler.fit(X)
        X=scaler.transform(X)
        inertia = []
        for i in range(1,11):
            kmeans = KMeans(
                n_clusters=i, init="k-means++",
                n_init=10,
                tol=1e-04, random_state=42
            )
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
        fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
                        yaxis={'title':'Inertia'},
                        annotations=[
                dict(
                    x=3,
                    y=inertia[2],
                    xref="x",
                    yref="y",
                    text="Elbow",
                    showarrow=True,
                    arrowhead=7,
                    ax=20,
                    ay=-40
                )
            ])

        test_kmean = KMeans(n_clusters=3)
        test_kmean.fit(X)
        y_km_test_kmean = test_kmean.fit_predict(X)
        test_cluster['cluster'] = y_km_test_kmean

        PLOTtest_kmean = go.Figure()

        for C in list(test_cluster.cluster.unique()):
            
            PLOTtest_kmean.add_trace(go.Scatter3d(x = test_cluster[test_cluster.cluster == C]['cases_new'],
                                        y = test_cluster[test_cluster.cluster == C]['total_test'],
                                        z = test_cluster[test_cluster.cluster == C]['pcr'],
                                        mode = 'markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(C)))
            

        PLOTtest_kmean.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                        scene = dict(xaxis=dict(title = 'cases new', titlefont_color = 'white'),
                                        yaxis=dict(title = 'total test', titlefont_color = 'white'),
                                        zaxis=dict(title = 'pcr', titlefont_color = 'white')),
                        font = dict(family = "Gilroy", color  = 'white', size = 12))

        j2.plotly_chart(PLOTtest_kmean,use_container_width=True)
        j2.plotly_chart(fig,use_container_width=True)
        st.write("From the 3d plot above we can see that we set the number of clusters to be 3 as the elbow indicates that the optimal number of cluster should be 3. To analyze the meaning of each cluster, we will also use a polar graph to help us to understand the characteristic of each cluster")
        k1,k2 =  st.columns((1,1))
        kmeans = KMeans(
        n_clusters=3, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
        )
        kmeans.fit(X)
        clusters=pd.DataFrame(X,columns=test_cluster.drop(["date","cluster"],axis=1).columns)
        clusters['label']=kmeans.labels_
        polar=clusters.groupby("label").mean().reset_index()
        polar=pd.melt(polar,id_vars=["label"])
        kmean_test = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=500,width=1400)
        k1.plotly_chart(kmean_test,use_container_width=True)
        k2.write("""
            We chose on using daily testing done (including rtk-ag and pcr) and daily new cases as our data for clustering because we think that if
            the country's daily testing is very important on reducing the risk of widespread of Covid-19 because if earlier detection of Covid-19 happened,
            we can immediately quarantine that person and his/her close contacts to prevent the disease from further spreading. To be clustered as cluster 2, the number
            of testing done daily should be very high and the number of daily cases should be very high. This cluster reflects the pandemic state of malaysia and has
            the meaning of "Severe State" because most of the testing results on that day are positive. 

            For cluster 1, the number of testing should be high but the number of positive case should be low. This cluster reflects the pandemic state of malaysia and has
            the meaning of "Slightly Risky State" because many testings are done on that day, but the number of positive results are low. It is slightly risky because
            there are still some positive cases and it still has the chance to cause an outbreak if proper epidemic prevention is not applied. 

            For cluster 0, the number of testing should be low and the number of positive cases should be low also. This cluster reflects the pandemic state of malaysia and has
            the meaning of "Risky State" because the number of testing done on that day is too low. Although low daily testing, but we can still see that there are some positive results
            among the testing. This could mean that there are still underlying cluster that has not been identified or ended yet hence it is risky to have such
            low total testing per day.

        """)
    with st.expander("DBScan"):
        l1,l2,l3 = st.columns((0.5,2,0.5))
        dbscan_test = DBSCAN(eps=0.3,min_samples = 10)
        dbscan_test.fit(X)
        test_cluster = test_msia.iloc[:, 0:4]
        test_cluster = test_cluster.dropna()
        test_cluster["total_test"] = test_cluster['rtk-ag'] + test_cluster['pcr']
        y_db_test = dbscan_test.labels_
        test_cluster['cluster'] = y_db_test


        PLOTdbscan_test = go.Figure()

        for C in list(test_cluster.cluster.unique()):
            
            PLOTdbscan_test.add_trace(go.Scatter3d(x = test_cluster[test_cluster.cluster == C]['cases_new'],
                                        y = test_cluster[test_cluster.cluster == C]['total_test'],
                                        z = test_cluster[test_cluster.cluster == C]['pcr'],
                                        mode = 'markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(C)))
            

        PLOTdbscan_test.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                        scene = dict(xaxis=dict(title = 'cases new', titlefont_color = 'white'),
                                        yaxis=dict(title = 'total test', titlefont_color = 'white'),
                                        zaxis=dict(title = 'pcr', titlefont_color = 'white')),
                        font = dict(family = "Gilroy", color  = 'white', size = 12))

        l2.plotly_chart(PLOTdbscan_test,use_container_width=True)
        st.write("From the 3d plot above we can see that the number of cluster identified by DBscan is only one. We are unable to find out meaning from the cluster as there are only one clusters being identified. This is probably due to the nature of the data and the algorithm of dbscan. The data is very near to each other hence DBscan will cluster all of them together. Hence DBScan is not suitable in this dataset.")
    with st.expander("Agglomerative Clustering"):
        m1,m2,m3 =  st.columns((0.5,2,0.5))
        dendo_test = ff.create_dendrogram(X)
        dendo_test.add_hline(y=1, line_width=3, line_dash="dash", line_color="green")
        dendo_test.update_layout(width=800, height=500)

        agg_test = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        agg_test.fit(X)
        agg_test_labels = agg_test.labels_

        test_cluster['cluster'] = agg_test_labels

        PLOTagg_test = go.Figure()

        for C in list(test_cluster.cluster.unique()):
            
            PLOTagg_test.add_trace(go.Scatter3d(x = test_cluster[test_cluster.cluster == C]['cases_new'],
                                        y = test_cluster[test_cluster.cluster == C]['total_test'],
                                        z = test_cluster[test_cluster.cluster == C]['pcr'],
                                        mode = 'markers', marker_size = 8, marker_line_width = 1,
                                        name = 'Cluster ' + str(C)))
            

        PLOTagg_test.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                        scene = dict(xaxis=dict(title = 'cases new', titlefont_color = 'white'),
                                        yaxis=dict(title = 'total test', titlefont_color = 'white'),
                                        zaxis=dict(title = 'pcr', titlefont_color = 'white')),
                        font = dict(family = "Gilroy", color  = 'white', size = 12))

        m2.plotly_chart(PLOTagg_test,use_container_width=True)
        m2.plotly_chart(dendo_test,use_container_width=True)
        clusters_agg_test= pd.DataFrame(X,columns= test_cluster.drop(["date","cluster"],axis=1).columns)
        clusters_agg_test['label']=agg_test.labels_
        polar_agg_test=clusters_agg_test.groupby("label").mean().reset_index()
        polar_agg_test=pd.melt(polar_agg_test,id_vars=["label"])
        fig_polar_agg_test = px.line_polar(polar_agg_test, r="value", theta="variable", color="label", line_close=True,height=500,width=800)
        n1,n2 = st.columns((1,1))
        n1.plotly_chart(fig_polar_agg_test,use_container_width=True)
        n2.write("""
            The polar plot of Agglomerative Clustering is identical to the polar plot of Kmean. Hence, our analysis will remain the same.

            From the polar plot, we can see that for to be classified as cluster 0, the number of check ins,unique location and unique individual of that day must be very high. This is because the main way for Covid-19 to spread is spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. A person can be infected when aerosols or droplets containing the virus are inhaled or come directly into contact with the eyes, nose, or mouth.
            With high number of check ins，unique location and unique individual, it shows that there are many places are crowded with different individuals. This scenario is perfect for an outbreak of Covid-19 to happen. Hence, we would like to say that cluster 0 brings the meaning of "very risky to cause widespread of covid-19".

            For cluster 2, it exhibits a similar properties as cluster 0. However it has a lower number of check ins. This shows that on that day, they are a moderate number of check ins at many places around malaysia with many different individuals. This condition also has the posibility to cause the widespread of covid-19, but it has fewer check ins meaning that 
            the number of people going out on that day is fewer. Hence, we would like to conclude that cluster 2 has the meaning of "risky to cause widespread of covid-19"

            For cluster 1, we can see that it has the least check ins, unique individual and unique location in a day. This shows that on that day, they are very less people heading out. This is an ideal setup
            to stop the widespread of Covid-19. Hence, we would like to say that cluster 1 has the meaning of "less risky to cause widespread of covid-19"
            """)

        st.markdown('#### Conclusion')
        st.markdown(""" ###### Both KMean and Agglomerative Clustering perform very well in identifying the cluster from the data. Both of them are able to quantify the risk of widespread of Covid-19 by using Check Ins, Unique Location and Unique Individual. """)
        st.write(" ")

    st.markdown('#### Conclusion for Clustering of Testing and Cases Data')
    st.markdown(""" ###### Both KMean and Agglomerative Clustering perform very well in identifying the cluster from the Testing and Cases dataset. Both of them are able to reflects the pandemic state of malaysia by using the daily total_test (rtk-ag + pcr) and daily new cases """)
    st.write(" ")
    ############ Streamlit ##############