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
import calendar
import altair as alt

def importCsv(name, listName): #function to import every csv in a folder
        dir_name = os.getcwd()
        for file in os.listdir(dir_name + '/'  +name):
            df = pd.read_csv(dir_name + '/'  + name +'/' +file, error_bad_lines=False)
            df.name = file
            listName.append(df)



def app():
    ########## Python ############
    

    pop = pd.read_csv('population.csv')
    epidemic_list = [] #declare list to store epidemic data
    cases_malaysia=pd.read_csv('epidemic data/cases_malaysia.csv')
    epidemic_list.append(cases_malaysia)
    cases_state=pd.read_csv('epidemic data/cases_state.csv')
    epidemic_list.append(cases_state)

    clusters=pd.read_csv('epidemic data/clusters.csv')
    epidemic_list.append(clusters)

    deaths_malaysia=pd.read_csv('epidemic data/deaths_malaysia.csv')
    epidemic_list.append(deaths_malaysia)

    deaths_state=pd.read_csv('epidemic data/deaths_state.csv')
    epidemic_list.append(deaths_state)

    hospital=pd.read_csv('epidemic data/hospital.csv')
    epidemic_list.append(hospital)

    icu=pd.read_csv('epidemic data/icu.csv')
    epidemic_list.append(icu)

    pkrc=pd.read_csv('epidemic data/pkrc.csv')
    epidemic_list.append(pkrc)

    tests_malaysia=pd.read_csv('epidemic data/tests_malaysia.csv')
    epidemic_list.append(tests_malaysia)

    tests_state=pd.read_csv('epidemic data/tests_state.csv')
    epidemic_list.append(tests_state)

    vaxcination_list = [] #declare list to store vaccination dataset
    vax_malaysia = pd.read_csv('vaxcination data/vax_malaysia.csv')
    vaxcination_list.append(vax_malaysia)
    vax_state = pd.read_csv('vaxcination data/vax_state.csv')
    vaxcination_list.append(vax_state)


    mysejahtera = [] #Declare list to store mysejahtera dataset
    checkin_malaysia = pd.read_csv('mysejahtera/checkin_malaysia.csv')
    mysejahtera.append(checkin_malaysia)
    checkin_malaysia_time = pd.DataFrame()
    mysejahtera.append(checkin_malaysia_time)
    checkin_state = pd.read_csv('mysejahtera/checkin_state.csv')
    mysejahtera.append(checkin_state)



    vax_msia = vaxcination_list[0]
    vax_state = vaxcination_list[1]
    vax_msia['date'] = pd.to_datetime(vax_msia['date'],errors='coerce') #change object to datetime
    cases_msia = epidemic_list[0]
    cases_state = epidemic_list[1]
    cases_msia['date'] = pd.to_datetime(cases_msia['date'],errors='coerce')
    vax_cases = vax_msia.merge(cases_msia,how='inner',on='date')
    vax_cases['date'] = pd.to_datetime(vax_cases['date'],errors='coerce')

    pfizer = pd.DataFrame()
    pfizer['date'] = vax_cases['date']
    pfizer['pfizer1'] = vax_cases['pfizer1']
    pfizer['pfizer2'] = vax_cases['pfizer2']

    az = pd.DataFrame()
    az['date'] = vax_cases['date']
    az['astra1'] = vax_cases['astra1']
    az['astra2'] = vax_cases['astra2']

    sinovac = pd.DataFrame()
    sinovac['date'] = vax_cases['date']
    sinovac['sinovac1'] = vax_cases['sinovac1']
    sinovac['sinovac2'] = vax_cases['sinovac2']

    cansino = pd.DataFrame()
    cansino['date'] = vax_cases['date']
    cansino['cansino'] = vax_cases['cansino']

    allvax = [pfizer,sinovac,az,cansino]
    allvax = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='outer'), allvax)

    allvax['Cumulative Partial'] = vax_cases['cumul_partial']
    allvax['Cumulative Fully'] = vax_cases['cumul_full']
    allvax['CasesOnThatDay'] = vax_cases['cases_new']
    allvax['Cases 7 days later'] = 0
    allvax['Cases 14 days later'] = 0
    
    print(allvax)
    for j in range(len(allvax)):
        date = allvax['date'][j] + relativedelta(weeks=+(1))
        after7days = allvax[allvax['date'] == date]
        if after7days.empty:
            allvax.iloc[[j],[11]] = 0
        else:
            allvax.iloc[[j],[11]] = after7days.CasesOnThatDay.values[0]
    
    for i in range(len(pfizer)):
        date = allvax['date'][i] + relativedelta(weeks=+(2))
        after14days = allvax[allvax['date'] == date]
        if after14days.empty:
            allvax.iloc[[i],[12]] = 0
        else:
            allvax.iloc[[i],[12]] = after14days.CasesOnThatDay.values[0]

    allvax['Fully'] = allvax['pfizer2'] + allvax['sinovac2'] + allvax['astra2'] + allvax['cansino']
    allvax['Cases Cumulative'] = allvax['CasesOnThatDay'].cumsum()
    allvax_melt = allvax.melt(id_vars='date',value_vars=['Fully',
        'CasesOnThatDay','Cases 7 days later','Cases 14 days later'])
    cum_melt = allvax.melt(id_vars='date',value_vars=['Cumulative Fully',
        'Cases Cumulative'])
    allvaxfig= px.line(allvax_melt, x='date', y="value",color='variable',line_shape='spline',title= "Time series plot of Daily Fully Vaccinated compared with cases on that day, cases on 7 days later and 14 days later")
    cum_meltfig = px.line(cum_melt, x='date', y="value",color='variable',line_shape='spline', title = "Total fully vaccinated vs Total Positive Covid-19 Cases")

    state_list = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
    'Pulau Pinang', 'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu',
    'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']

    deaths_state = epidemic_list[4]
    death_statedf = []
    for x in state_list:
        death_statedf.append(deaths_state[deaths_state['state'] == x].reset_index())


    cases_statedf = []
    for x in state_list:
        cases_statedf.append(cases_state[cases_state['state'] == x].reset_index())

    def percentInfected(state,pop):
        percent = state / pop
        return percent * 100

    def testingRate(state,days):
        rate = state/days
        return rate

    percentInfectedList = []
    i = 1
    j = 0
    for x in cases_statedf:
        state_name = x['state'][0]
        state_death = death_statedf[j]
        percentInfectedList.append([state_name,percentInfected(x['cases_active'].iloc[-1],pop['pop'][i]),x['cases_active'].iloc[-1],x['cases_new'].sum(),x['cases_recovered'].sum(),state_death['deaths_new'].sum(),pop['pop'][i]])
        i += 1
        j += 1


    state_infected = pd.DataFrame(data =percentInfectedList,columns=['State','Infected Percentage','Cases Active (Total)','Total Infected','Total Recovered','Total Death','Population'] )
    state_infected.sort_values(by='Infected Percentage',ascending = False).reset_index().drop('index',axis=1)
    fig_percentInfected = px.bar(state_infected.sort_values(by='Infected Percentage',ascending = False).reset_index().drop('index',axis=1), x='State', y='Infected Percentage', title="Percent of population currently infected with COVID-19 of each state")

    vax_statedf = []
    for x in state_list:
        vax_statedf.append(vax_state[vax_state['state'] == x].reset_index())


    percentVaccinatedList = []
    i = 1 
    for x in vax_statedf:
        percentVaccinatedList.append([x['state'][0],percentInfected(x['cumul_full'].iloc[-1],pop['pop'][i]),pop['pop'][i]])
        i += 1

    state_vaccinated = pd.DataFrame(data =percentVaccinatedList,columns=['State','Vaccinated Percentage','Population'] )

    fig_state_vaccinated = px.bar(state_vaccinated.sort_values(by='Vaccinated Percentage',ascending = True).reset_index().drop('index',axis=1)  , x='State', y='Vaccinated Percentage', title="Percent of population vaccinated of each state")

    test_state = epidemic_list[9]
    test_state.shape[0]

    test_statedf = []
    for x in state_list:
        test_statedf.append(test_state[test_state['state'] == x].reset_index())

    TestingRateList = []
    i = 1 
    j = 0
    for x in test_statedf:
        TestingRateList.append([x['state'][0],testingRate(x['rtk-ag'].sum() + x['pcr'].sum(),x.shape[0]), x['rtk-ag'].sum()+ x['pcr'].sum(),state_infected['Total Infected'][j]])
        i += 1
        j += 1

    state_testingRate = pd.DataFrame(data =TestingRateList,columns=['State','Average Testing Rate per Day','Total Test Done','Total Infected'] )
    fig_state_testingRate = px.bar(state_testingRate.sort_values(by='Average Testing Rate per Day',ascending = True).reset_index().drop('index',axis=1)  , x='State', y='Average Testing Rate per Day', title="Testing Rate of each state")

    move = pd.read_csv("Mysejahtera/checkin_malaysia.csv")
    
    ########## Python ############



    ############ Streamlit ##############
    st.markdown('## Exploratory Data Analysis')
    st.write("In this section, we will be exploring the datasets provided by MOH and try to answers some data science questions to bring meaningful insight.")
    st.write(" ")
    st.markdown('### Has vaccination helped reduced the daily cases?')
    c1,c2 = st.columns((2,1))
    c1.plotly_chart(allvaxfig,use_container_width=True)
    c2.write(""" 
        For this question, we consider people who are fully vaccinated only which is indicated as "Fully" in the line plot above. Our idea is that, 
        we are trying to compare the cases on that day and cases after 7 and 14 days (Number of days required for the covid-19 vaccine to be fully effective)
        with the number of fully vaccinated person on that specific day. From the plot above, it can be clearly seen that the number of fully vaccinated person
        increase a lot everyday but we do not see much changes in the daily cases. Hence from this plot we could not conclude whether vaccination helps
        in reducing daily cases as there might be simpson paradox like the movement control order which definitely prevent the spread of covid-19. 
    """)
    d1,d2 = st.columns((2,1))
    d1.plotly_chart(cum_meltfig,use_container_width=True)
    d2.write("""
        From the plot of Total fully vaccinated vs Total Positive Covid-19 Cases, we can see that the total number of fully vaccinated citizens is 20 million and the total number of positive Covid-19 cases is nearly 2 million. Malaysia has an estimated population of 30 million, 20 million of them are adults and the total of fully vaccinated population are all adults. Most of the Covid-19 cases are infected around workplaces. We could not conclude on whether covid-19 vaccine helps in reducing daily cases because as you can see        from the graph, the cumulative cases has a spike around August 2021 despite having an enormous difference with the total number of fully vaccinated persons. On top of that, although the cumulative        cases are growing at a very slow pace compared with cumulative fully vaccinated people, we could not conclude that vaccination is suppressing the growth of cumulative cases. This is because movement control orders issued by the government are also one of the plans by the government to reduce daily cases (simpson paradox). However, the movement control issued by the government is ended as most of the adults are vaccinated, we might be able to conclude the effectiveness of vaccination after a few months as vaccines are the only thing protecting the citizens from covid-19. 
    """)
    st.markdown('##### Conclusion')
    st.markdown('###### We could not make a conclusion on vaccination does helps in reducing daily cases as they are underlying factors like movement control order which are also a plan by the government to reduce daily cases. We could not just conclude that vaccination is the plan that are suppressing the daily cases as MCO might be also helping.')
    st.write(" ")
        
    st.markdown('### What state(s) require attention now?')
    st.write("There are many aspect in which a state can be given attention. Hence, we have decided to measure the seriousness of COVID-19 in each state by the following aspects:")
    st.write("1.) Percent of population infected with COVID-19 of a state")    
    st.write("2.) Percent of population vaccinated of a state")
    st.write("3.) The testing rate of a state")
    st.write("")
    
    e1,e2 = st.columns((1,1))
    e1.markdown('##### First Aspect: Percent of population infected with COVID-19 of a state')
    e1.write("""
    We decided to look at the seriousness of COVID-19 in each state using percent of population currently infected with COVID-19 because if the infected percentage is too high, this will heavily impact the life of people living there as most of them will likely be infected by Covid-19. The economy of the state will be affected    too as most of the people would not dare to go out and work or spend money. Moreover, if most of them are infected there will be many deaths that follow. Hence, those states that have a higher infected percentage, should be given more attention.
    """)
    e1.plotly_chart(fig_percentInfected,use_container_width=True)
    e1.write(""" From the bar chart, we can see that Sarawak has the highest infected percentage among all the states, which is 2.18%. Sarawak requires more attention now to prevent more of their citizens being infected. If the infected percentage continues to increase, it will heavily impact the economic activities in Sarawak and the citizens will suffer. To prevent this from happening, the government should gives more attention to Sarawak in preventing Covid-19 to spread 
    """)

    e2.markdown('##### Second Aspect: Percent of vaccinated population of a state')
    e2.write("""
    We decided to look at the seriousness of COVID-19 in each state using percent of the vaccinated population because COVID-19 is deadly to those who are not vaccinated. If covid-19 outbreaks, it will be deadly to those states which have a very low percent of vaccinated population and it will also heavily impact the economic activities inside that state. Hence, if the percent of vaccinated population of a state is very low, the government should give more attention to that state to prevent the people living there dying from covid-19 by increasing the number of vaccinations given in a day. 
    """)
    e2.plotly_chart(fig_state_vaccinated,use_container_width=True)
    
    e2.write(""" From the bar chart, we can see that Terengganu and Sabah have very low percent of the vaccinated population among all the states, which is 24.78% and 26.58% respectively. Terengganu and Sabah require more attention now to prevent covid-19 to be deadly to the people living there. The government should focus more on encouraging and giving vaccination to the people living in both Sabah and Terengganu as vaccination is proven to protect humans from  covid-19.
    """)
    
    st.write(" ")
    f1,f2,f3 =  st.columns((0.5,1,0.5))
    f2.markdown("#### Third Aspect: The testing rate of a state")
    f2.write("""
        We decided to look at the seriousness of COVID-19 in each state using testing rate because if the average testing rate per day of a state is very low, the state has a higher possibility that there are undetected covid-19 clusters around the citizens. This could be serious as it might cause an outbreak which will lead to many unforeseen circumstances. Hence, states that have relatively low testing rate should be given more attention by doing more testing in a day.
    """)

    f2.plotly_chart(fig_state_testingRate,use_container_width=True)

    f2.write("""
        From the bar chart above, we can see that Perlis, Putrajaya and Labuan have very low testing rate this is probably because they have very low daily cases. However,they should not be neglected. They should be given more attention if there are still some cases in those states because there might be undetected covid-19 clusters around the citizens since the testing rate is so low.

    """)

    st.markdown('#### Conclusion')
    st.markdown("###### All state should be given the same amount of attention so that the people would not suffer from any outbreaks of Covid-19 as it will heavily impact the economy. However if we consider in the aspects that we mentioned above, Sarawak should be given more attention if according to percent of population infected with Covid-19, Terengganu and Sabah should be given more attention if according to percent of vaccinated population and Perlis, Putrajaya and Labuan should be given more attention if according to the testing rate.")
    st.write(" ")

    st.markdown('### Does the current vaccination rate allow herd immunity to be achieved by 30 November 2021?')
    with st.expander("See Explaination"):
        st.write("We assume that herd immunity can be achieved with 80% of the population having been vaccinated. The approach we used to solve this question is using simple mathematics and the steps is stated in below.")
        st.write("Step 1: We find out the percentage of vaccinated people in Malaysia which is 63.75%.")    
        st.write("Step 2: We calculate how many more people need vaccination to achieve 80% which is 5,306,666.")
        st.write("Step 3: We calculate the average rate of vaccination which is 199990.73 vaccination per day.")
        st.write("Step 4: We calculate how many people will be vaccinated on november 30th based on the rate of vaccination. We will have 32018736 vaccinated people by november 30th which is 98.044%.Therefore we can reach herd immunity.")
        st.write(" ")
        st.write("However, the approach we use to solve this problem does not take into account other factors such as willingness of people to get vaccinated, supply of vaccination and other underlying factors. ")
        st.markdown('#### Conclusion')
        st.write("By utilizing simple mathematics, we conclude that Malaysia will have 32018736 vaccinated people by november 30th which is 98.044%. Therefore we can reach herd immunity by 30 November 2021.")

    st.markdown('### What is the difference of covid trend in 2020 and 2021 ?')
    cases_msia = epidemic_list[0]
    cases_msia['day'] = cases_msia['date'].dt.day
    cases_msia['month'] = cases_msia['date'].dt.month
    cases_msia['year'] = cases_msia['date'].dt.year

    year2020 = cases_msia['year'] == 2020
    cases_msia_2020 = cases_msia[year2020] 

    year2021 = cases_msia['year'] == 2021
    cases_msia_2021 = cases_msia[year2021]

    cases_msia_2020_byMth = cases_msia_2020.groupby(['month'])['cases_new'].sum().reset_index()
    cases_msia_2020_byMth['month'] = cases_msia_2020_byMth['month'] .apply(lambda x: calendar.month_name[x])
    cases_msia_2020_byMth

    cases_msia_2021_byMth = cases_msia_2021.groupby(['month'])['cases_new'].sum().reset_index()
    cases_msia_2021_byMth['month'] = cases_msia_2021_byMth['month'] .apply(lambda x: calendar.month_name[x])
    cases_msia_2021_byMth = cases_msia_2021_byMth.append([{'month' : 'October'},{'month' : 'November'},{'month' : 'December'}], ignore_index = True)

    cases_varies = cases_msia_2020_byMth.merge(cases_msia_2021_byMth,on='month')
    cases_varies.columns = ['Month','Year 2020 Cases','Year 2021 Cases']
    
    cases_melt = cases_varies.melt(id_vars='Month',value_vars=['Year 2020 Cases','Year 2021 Cases'])

    n1,n2 = st.columns((2,1))
    n1.plotly_chart(px.line(cases_melt,x='Month',y = 'value',title='Cases of 2020 vs Cases of 2021',color='variable'),use_container_width=True)
    n2.write("""From this graph we can see that monthly cases of 2020 relative to monthly cases in 2021 is quite low. We can see that the monthly case in august 2021 reached a peak of over 600,000 cases and started to go down in september. It goes even lower in October but this is due to insufficient data as we only have data till oct 5. From this chart we can see the state of the pandemic. For example we can see that from June to August there was a huge increase in cases indicating that the covid cases are getting worse. However from August to September it slowly decreased and possibly even decreased more in October but we cannot conclude that it does really go down in October due to the data set only having a few days data of october. """)

    st.markdown('### Trend of different cluster types in 2021')
    value = ['cluster_import',	'cluster_religious',	'cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace']

    t = cases_msia_2021.groupby(['month'])['cluster_import',	'cluster_religious',	'cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace'].sum().reset_index()
    t_cases_melt = t.melt(id_vars='month',value_vars=['cluster_import',	'cluster_religious',	'cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace'])

    o1,o2 = st.columns((2,1))
    o1.plotly_chart(px.line(t_cases_melt, x='month', y='value', color='variable'),use_container_width=True)
    o2.write("""In this question we see the trends of different cluster types such as import, religious, community, detention center, education and workplace in 2021. We only see the cluster in 2021 because in 2020 all these clusters were null.
     From the chart we can see which cluster type is responsible for the highest number of covid cases in 2021.""")

    st.markdown('### Is there any correlation between vaccination and daily cases for all states of Malaysia?')
    st.write("""
        We look at this question from two different perspective, correlation between vaccination and daily cases for all states of Malaysia; 
        correlation between vaccination and daily cases for all states of Malaysia, starting from 26-02-2021.  Before 26-02-2021, Malaysia is
        has not given any vaccination because the vaccines have not arrived yet. 
    """)
    vax_state = vaxcination_list[1]
    cases_state = epidemic_list[1]
    testing = cases_state.merge(vax_state, how='left', on=['date','state']) # merging vaccination by state data and cases by state data
    testing = testing.fillna(0)
    st.markdown('#### Correlation between vaccination and daily cases for all states of Malaysia (Including Date Before 26-02-2021)')
    state_list = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
    'Pulau Pinang', 'Perak', 'Perlis', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu',
    'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']
    charts = []
    for x in state_list: # plot scatter plot using altair by states
        y = testing[testing['state']== x]
        chart = alt.Chart(y,title=x+ ' '+(y['daily'].corr(y['cases_new'])).astype(str)).mark_point().encode(
            x='daily',
            y='cases_new'
        ).interactive()
        charts.append(chart)
    with st.expander("Show correlation plot of each states"):
        p1,p2,p3,p4 =  st.columns((1,1,1,1))
        p1.write("Johor")
        p1.altair_chart(charts[0],use_container_width=True)
        p2.write("Kedah")
        p2.altair_chart(charts[1],use_container_width=True)
        p3.write("Kelantan")
        p3.altair_chart(charts[2],use_container_width=True)
        p4.write("Melaka")
        p4.altair_chart(charts[3],use_container_width=True)

        p1.write("Negeri Sembilan")
        p1.altair_chart(charts[4],use_container_width=True)
        p2.write("Pahang")
        p2.altair_chart(charts[5],use_container_width=True)
        p3.write("Pulau Pinang")
        p3.altair_chart(charts[6],use_container_width=True)
        p4.write("Perak")
        p4.altair_chart(charts[7],use_container_width=True)

        p1.write("Perlis")
        p1.altair_chart(charts[8],use_container_width=True)
        p2.write("Sabah")
        p2.altair_chart(charts[9],use_container_width=True)
        p3.write("Sarawak")
        p3.altair_chart(charts[10],use_container_width=True)
        p4.write("Selangor")
        p4.altair_chart(charts[11],use_container_width=True)

        p1.write("Terrenganu")
        p1.altair_chart(charts[12],use_container_width=True)
        p2.write("Kuala Lumpur")
        p2.altair_chart(charts[13],use_container_width=True)
        p3.write("Labuan")
        p3.altair_chart(charts[14],use_container_width=True)
        p4.write("Putrajaya")
        p4.altair_chart(charts[15],use_container_width=True)

    st.markdown('#### Correlation between vaccination and daily cases for all states of Malaysia (Excluding Date Before 26-02-2021)')
    testing = testing[testing.date >= '2021-02-26'] #selecting date from 2021-02-26 to current date
    chartsafter = []
    for x in state_list: #plot scatter plot based on state using altair
        y = testing[testing['state']== x]
        chart = alt.Chart(y,title=x+ ' '+(y['daily'].corr(y['cases_new'])).astype(str)).mark_point().encode(
            x='daily',
            y='cases_new'
        ).interactive()
        chartsafter.append(chart)
    with st.expander("Show correlation plot of each states"):
        q1,q2,q3,q4 =  st.columns((1,1,1,1))
        q1.write("Johor")
        q1.altair_chart(chartsafter[0],use_container_width=True)
        q2.write("Kedah")
        q2.altair_chart(chartsafter[1],use_container_width=True)
        q3.write("Kelantan")
        q3.altair_chart(chartsafter[2],use_container_width=True)
        q4.write("Melaka")
        q4.altair_chart(chartsafter[3],use_container_width=True)

        q1.write("Negeri Sembilan")
        q1.altair_chart(chartsafter[4],use_container_width=True)
        q2.write("Pahang")
        q2.altair_chart(chartsafter[5],use_container_width=True)
        q3.write("Pulau Pinang")
        q3.altair_chart(chartsafter[6],use_container_width=True)
        q4.write("Perak")
        q4.altair_chart(chartsafter[7],use_container_width=True)

        q1.write("Perlis")
        q1.altair_chart(chartsafter[8],use_container_width=True)
        q2.write("Sabah")
        q2.altair_chart(chartsafter[9],use_container_width=True)
        q3.write("Sarawak")
        q3.altair_chart(chartsafter[10],use_container_width=True)
        q4.write("Selangor")
        q4.altair_chart(chartsafter[11],use_container_width=True)

        q1.write("Terrenganu")
        q1.altair_chart(chartsafter[12],use_container_width=True)
        q2.write("Kuala Lumpur")
        q2.altair_chart(chartsafter[13],use_container_width=True)
        q3.write("Labuan")
        q3.altair_chart(chartsafter[14],use_container_width=True)
        q4.write("Putrajaya")
        q4.altair_chart(chartsafter[15],use_container_width=True)

    st.write("")
    st.write("")

    st.markdown('##### Conclusion')
    st.write("")
    ti1,ti2 = st.columns((1,1))
    ti1.markdown('__Correlation between vaccination and daily cases for all states of Malaysia (Including Date Before 26-02-2021)__')
    ti2.markdown('__Correlation between vaccination and daily cases for all states of Malaysia (Excluding Date Before 26-02-2021)__')

    r1,r2,r3,r4= st.columns((1,1,1,1))
    r1.markdown("<h6 style='text-align: center'>Strength of correlation</h6>", unsafe_allow_html=True)
    r2.markdown("<h6 style='text-align: center'>Number of states</h6>", unsafe_allow_html=True)
    r1.markdown("<p style='text-align: center'>Strong (0.7 - 1)</p>", unsafe_allow_html=True)
    r2.markdown("<p style='text-align: center'>14</p>", unsafe_allow_html=True)
    r1.markdown("<p style='text-align: center'>Moderate (0.5 - 0.69)</p>", unsafe_allow_html=True)
    r2.markdown("<p style='text-align: center'>1</p>", unsafe_allow_html=True)
    r1.markdown("<p style='text-align: center'>Weak (0.0 - 0.49)</p>", unsafe_allow_html=True)
    r2.markdown("<p style='text-align: center'>2</p>", unsafe_allow_html=True)

    r3.markdown("<h6 style='text-align: center'>Strength of correlation</h6>", unsafe_allow_html=True)
    r4.markdown("<h6 style='text-align: center'>Number of states</h6>", unsafe_allow_html=True)
    r3.markdown("<p style='text-align: center'>Strong (0.7 - 1)</p>", unsafe_allow_html=True)
    r4.markdown("<p style='text-align: center'>13</p>", unsafe_allow_html=True)
    r3.markdown("<p style='text-align: center'>Moderate (0.5 - 0.69)</p>", unsafe_allow_html=True)
    r4.markdown("<p style='text-align: center'>1</p>", unsafe_allow_html=True)
    r3.markdown("<p style='text-align: center'>Weak (0.0 - 0.49)</p>", unsafe_allow_html=True)
    r4.markdown("<p style='text-align: center'>3</p>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown("###### In this question, we did a correlation check for vaccination and daily cases. We first check the overall correlation of daily vaccination data and daily covid cases data. We can see that in both cases the majority of the states exhibit positive strong correlation and only some states exhibit moderate and weak positive correlation. However we cannot make a conclusion that increase in vaccination numbers will cause an increase in daily covid cases as there could be underlying factors that we have not taken into account.")
    st.markdown("###### Next we check correlation of daily vaccination and daily covid cases from 26-02-2021 onwards. We chose to check from this date onwards because the first batch of the vaccine was released on this date. So we attempted to check the correlation to see if there is a difference. However, the result we found was similar to the previous one. We can see a strong positive correlation in the majority of the states but we cannot conclude the causality between these two variables for the same reason as above.")
    st.write("")

    st.markdown('### How close are the states from achieving herd immunity (80%)')
    st.write("To answer this question we will use simple mathematics to calculate percent of vaccinated by taking the population of each state and divide by the cumulative number of vaccinations in that state. The vaccinated percentage of each state is shown in the bar plot below.")

    tempdf = pd.DataFrame(columns =['state','number fully vaccinated', 'population', 'percent vaccinated']) #create new data frame with these columns
    for x in state_list: #filling the new dataframe with data by states
        y = vax_state[vax_state['state']==x] #vax data filter by state
        z =pop[pop['state'] == x]['pop'] #pop of the state
        
        percentVax = y.iloc[-1]['cumul_full'] / z *100 #calculate percent vaccinated
        dicttemp = {'state': x, 'number fully vaccinated': y.iloc[-1]['cumul_full'], 'population': z.values[0], 'percent vaccinated':percentVax.values[0]}
        tempdf = tempdf.append(dicttemp, ignore_index = True)

    figq8 = go.Figure(go.Bar(
            y=tempdf['state'],
            x=tempdf['percent vaccinated'],
            orientation='h'))

    figq8.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=3
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Percent Vaccinated %",
        yaxis_title="States",
    )

    st.plotly_chart(figq8,use_container_width=True)
    st.markdown('##### Conclusion')
    st.markdown("###### We found that all states were at least half way there to achieve herd immunity of 80%. We also found that Kuala Lumpur and Putrajaya had more than 100%. This may be caused by the lack of understanding on whether the number of fully vaccinated is counted according to their citizen or they just counted based on the number of people that received their second dose on that day. If it is counted based on the number of people that received their second dose on that day without regard of whether that particular person is from that state, this will causes the number of fully vaccinated to be larger than the state population hence causing the percentage to be larger than 100%.")

    st.markdown('### Does high number of check in result in increase of covid cases?')
    checkin_msia = mysejahtera[0]
    checkin_state = mysejahtera[2]
    cases_msiaDated = cases_msia[cases_msia['date'] >= '2020-12-01'] #selecting cases_msia from 2020-12-01 onwards
    checkin_msia['date'] =  pd.to_datetime(checkin_msia['date'])
    dfCheckinCases = checkin_msia.merge(cases_msiaDated, on='date',how='left')[['date','checkins','cases_active']] #merge checkin state data and cases msia

    fig9 = go.Figure()

# Add traces
    fig9.add_trace(go.Scatter(x=dfCheckinCases['date'], y=dfCheckinCases['checkins'],
                        mode='lines',
                        name='checkins'))
    fig9.add_trace(go.Scatter(x=dfCheckinCases['date'], y=dfCheckinCases['cases_active'],
                        mode='lines',
                        name='cases_new'))


    fig9.update_layout(
        xaxis_title="Cases New (Daily)",
        yaxis_title="Check-in (Daily)",
        title="Cases New (Daily) vs Check-in (Daily)",
    )

    st.plotly_chart(fig9,use_container_width=True)
    st.markdown('##### Conclusion')
    st.markdown("###### From the plot above we can see that although there huge ups and downs in daily total number of check-in, the daily cases still remain steady. We also did a correlation check using and the result is only __0.08__. It show no correlation between number of checkins and cases active.")






    
    
    




    
        
        
   