import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup
import re
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from operator import itemgetter

dflist = []
# get the response in the form of html
wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)_per_capita"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
gdptable=soup.find('table',{'class':"wikitable"})
string = str(gdptable)
gdpdf=pd.read_html(string)
gdpdf=pd.DataFrame(gdpdf[0])
gdpdf = gdpdf.drop(gdpdf.columns[[1,2,3,4,5,7]], axis=1)
gdpdf.columns = ["Country", "GDP_Estimate"]
gdpdf["GDP_Estimate"]= pd.to_numeric(gdpdf["GDP_Estimate"],errors='coerce')
gdpdf = gdpdf.drop(gdpdf.index[0])
print(gdpdf["GDP_Estimate"].mean())
print(gdpdf["GDP_Estimate"].median())
print(gdpdf["GDP_Estimate"].max())
print(gdpdf.mode()['GDP_Estimate'][0])
print(gdpdf["GDP_Estimate"].min())
print(gdpdf["GDP_Estimate"].var())
print(gdpdf["GDP_Estimate"].isna().sum()) 
dflist.append(gdpdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_Internet_connection_speeds"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
internettable=soup.find('table',{'class':"wikitable"})
string = str(internettable)
internetdf=pd.read_html(string)
internetdf=pd.DataFrame(internetdf[0])
internetdf = internetdf.drop(internetdf.columns[[0,3,4]], axis=1)
internetdf.columns = ["Country", "InternetSpeed"]
internetdf["InternetSpeed"]= pd.to_numeric(internetdf["InternetSpeed"],errors='coerce')
print(internetdf["InternetSpeed"].mean())
print(internetdf["InternetSpeed"].median())
print(internetdf["InternetSpeed"].max())
print(internetdf.mode()["InternetSpeed"][0])
print(internetdf["InternetSpeed"].min())
print(internetdf["InternetSpeed"].var())
print(internetdf["InternetSpeed"].isna().sum()) 
dflist.append(internetdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_alcohol_consumption_per_capita"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find('table',{'class':"wikitable"})
string = str(table)
alcoholdf=pd.read_html(string)
alcoholdf=pd.DataFrame(alcoholdf[0])
alcoholdf = alcoholdf.drop(alcoholdf.columns[[1]], axis=1)
alcoholdf.columns = ["Country", "Alcohol"]
alcoholdf["Alcohol"]= pd.to_numeric(alcoholdf["Alcohol"],errors='coerce')
print(alcoholdf["Alcohol"].mean())
print(alcoholdf["Alcohol"].median())
print(alcoholdf["Alcohol"].max())
print(alcoholdf.mode()["Alcohol"][0])
print(alcoholdf["Alcohol"].min())
print(alcoholdf["Alcohol"].var())
print(alcoholdf["Alcohol"].isna().sum())
dflist.append(alcoholdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_intentional_homicide_rate#By_country,_region,_or_dependent_territory"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
homicidedf=pd.read_html(string)
homicidedf=pd.DataFrame(homicidedf[1])
homicidedf = homicidedf.drop(homicidedf.columns[[1,2,4,5,6]], axis=1)
homicidedf.columns = ["Country", "Homicide"]
homicidedf["Homicide"]= pd.to_numeric(homicidedf["Homicide"],errors='coerce')
homicidedf = homicidedf.drop(homicidedf.index[0])
print(homicidedf["Homicide"].mean())
print(homicidedf["Homicide"].median())
print(homicidedf["Homicide"].max())
print(homicidedf.mode()["Homicide"][0])
print(homicidedf["Homicide"].min())
print(homicidedf["Homicide"].var())
print(homicidedf["Homicide"].isna().sum())
dflist.append(homicidedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_military_expenditures"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find('table',{'class':"wikitable"})
string = str(table)
militarydf=pd.read_html(string)
militarydf=pd.DataFrame(militarydf[0])
militarydf = militarydf.drop(militarydf.columns[[0,2,4]], axis=1)
militarydf.columns = ["Country", "MilitarySpending"]
militarydf["MilitarySpending"]= pd.to_numeric(militarydf["MilitarySpending"],errors='coerce')
militarydf = militarydf.drop(militarydf.index[0])
print(militarydf["MilitarySpending"].mean())
print(militarydf["MilitarySpending"].median())
print(militarydf["MilitarySpending"].max())
print(militarydf.mode()["MilitarySpending"][0])
print(militarydf["MilitarySpending"].min())
print(militarydf["MilitarySpending"].var())
print(militarydf["MilitarySpending"].isna().sum())
dflist.append(militarydf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_Human_Development_Index"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
humandf=pd.read_html(string)
humandf=pd.DataFrame(humandf[1])
humandf = humandf.drop(humandf.columns[[0,1,4]], axis=1)
humandf.columns = ["Country", "Human_Index"]
humandf["Human_Index"]= pd.to_numeric(humandf["Human_Index"],errors='coerce')
print(humandf["Human_Index"].mean())
print(humandf["Human_Index"].median())
print(humandf["Human_Index"].max())
print(humandf.mode()["Human_Index"][0])
print(humandf["Human_Index"].min())
print(humandf["Human_Index"].var())
print(humandf["Human_Index"].isna().sum()) 
dflist.append(humandf)


wikiurl="https://en.wikipedia.org/wiki/Democracy_Index"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
democracydf=pd.read_html(string)
democracydf=pd.DataFrame(democracydf[3])
democracydf = democracydf.drop(democracydf.columns[[0,1,3,4,6,7,8,9,10,11,12,13,14,15,16,17]], axis=1)
democracydf.columns = ["Country", "DemocracyIndex"]
democracydf["DemocracyIndex"]= pd.to_numeric(democracydf["DemocracyIndex"],errors='coerce')
print(democracydf["DemocracyIndex"].mean())
print(democracydf["DemocracyIndex"].median())
print(democracydf["DemocracyIndex"].max())
print(democracydf.mode()["DemocracyIndex"][0])
print(democracydf["DemocracyIndex"].min())
print(democracydf["DemocracyIndex"].var())
print(democracydf["DemocracyIndex"].isna().sum())
dflist.append(democracydf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_tertiary_education_attainment"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
educationdf=pd.read_html(string)
educationdf=pd.DataFrame(educationdf[1])
educationdf = educationdf.drop(educationdf.columns[[2,3,4,5]], axis=1)
educationdf.columns = ["Country", "Education"]
educationdf["Education"]= pd.to_numeric(educationdf["Education"],errors='coerce')
print(educationdf["Education"].mean())
print(educationdf["Education"].median())
print(educationdf["Education"].max())
print(educationdf.mode()["Education"][0])
print(educationdf["Education"].min())
print(educationdf["Education"].var())
print(educationdf["Education"].isna().sum()) 
dflist.append(educationdf)


wikiurl="https://en.wikipedia.org/wiki/Importance_of_religion_by_country"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find('table',{'class':"wikitable"})
string = str(table)
religiondf=pd.read_html(string)
religiondf=pd.DataFrame(religiondf[0])
religiondf = religiondf.drop(religiondf.columns[[0,3]], axis=1)
religiondf.columns = ["Country", "Religion"]
religiondf['Religion'] = religiondf['Religion'].str.replace('%', '',regex=True)
religiondf['Religion'] = religiondf['Religion'].str.replace('\[1\]', '',regex=True)
religiondf["Religion"]= pd.to_numeric(religiondf["Religion"],errors='coerce')
print(religiondf["Religion"].mean())
print(religiondf["Religion"].median())
print(religiondf["Religion"].max())
print(religiondf.mode()["Religion"][0])
print(religiondf["Religion"].min())
print(religiondf["Religion"].var())
print(religiondf["Religion"].isna().sum()) 
dflist.append(religiondf)


wikiurl="https://en.wikipedia.org/wiki/Christianity_by_country"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
christiandf=pd.read_html(string)
christiandf=pd.DataFrame(christiandf[2])
christiandf = christiandf.drop(christiandf.columns[[1,3,4]], axis=1)
christiandf.columns = ["Country", "Christianity"]
christiandf = christiandf.drop(christiandf.index[[-1,-2,-3,-4,-5,-6,-7,-8]])
christiandf['Christianity'] = christiandf['Christianity'].str.replace('%', '')
christiandf['Christianity'] = christiandf['Christianity'].str.replace('\[[0-9]+\]', '', regex=True)
christiandf["Christianity"]= pd.to_numeric(christiandf["Christianity"],errors='coerce')
print(christiandf["Christianity"].mean())
print(christiandf["Christianity"].median())
print(christiandf["Christianity"].max())
print(christiandf.mode()["Christianity"][0])
print(christiandf["Christianity"].min())
print(christiandf["Christianity"].var())
print(christiandf["Christianity"].isna().sum())
dflist.append(christiandf)


wikiurl="https://en.wikipedia.org/wiki/Islam_by_country"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
islamdf=pd.read_html(string)
islamdf=pd.DataFrame(islamdf[0])
islamdf = islamdf.drop(islamdf.columns[[1,2,4,5]], axis=1)
islamdf.columns = ["Country", "Islam"]
islamdf['Islam'] = islamdf['Islam'].str.replace('<', '')
islamdf["Islam"]= pd.to_numeric(islamdf["Islam"],errors='coerce')
print(islamdf["Islam"].mean())
print(islamdf["Islam"].median())
print(islamdf["Islam"].max())
print(islamdf.mode()["Islam"][0])
print(islamdf["Islam"].min())
print(islamdf["Islam"].var())
print(islamdf["Islam"].isna().sum())
dflist.append(islamdf)


wikiurl="https://en.wikipedia.org/wiki/Buddhism_by_country"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
buddhismdf=pd.read_html(string)
buddhismdf=pd.DataFrame(buddhismdf[0])
buddhismdf = buddhismdf.drop(buddhismdf.columns[[1,2,3,4,5,7,8]], axis=1)
buddhismdf.columns = ["Country", "Buddhism"]
buddhismdf['Buddhism'] = buddhismdf['Buddhism'].str.replace('<', '')
buddhismdf['Buddhism'] = buddhismdf['Buddhism'].str.replace('%', '')
buddhismdf["Buddhism"]= pd.to_numeric(buddhismdf["Buddhism"],errors='coerce')
print(buddhismdf["Buddhism"].mean())
print(buddhismdf["Buddhism"].median())
print(buddhismdf["Buddhism"].max())
print(buddhismdf.mode()["Buddhism"][0])
print(buddhismdf["Buddhism"].min())
print(buddhismdf["Buddhism"].var())
print(buddhismdf["Buddhism"].isna().sum())
dflist.append(buddhismdf)


wikiurl="https://en.wikipedia.org/wiki/Jewish_population_by_country"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
jewishdf=pd.read_html(string)
jewishdf=pd.DataFrame(jewishdf[0])
jewishdf = jewishdf.drop(jewishdf.columns[[1,2,4,5,6,7,8,9,10,11,12,13]], axis=1)
jewishdf.columns = ["Country", "Jewish"]
jewishdf["Jewish"]= [43,39,3,2.9,2.7,2,1.2,1,0.8,0.8,0.62,0.35,0.32,0.3,0.27,0.2,0.2,0.18,0.13,0.11,0.11,0.1,0.098,0.088,0.07,0.068,0.064,0.06,0.057,0.051,0.049,0.043,0.041,0.032,0.03,0.03,0.028,0.026,0.021,0.02,0.02,0.018,0.018,0.017,0.017,0.016,0.014,0.014,0.014,0.013,0.013,0.013,0.011,0.01,0.01,0.0095,0.0088,0.0088,0.0074,0.0068,0.0068,0.0061,0.0061,0.0054,0.0047,0.0041,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0034,0.0027,0.0027,0.002,0.0014,0.0014,0.0014,0.0014,0.0014,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,0.00068,"—","—","—","—","—","—","—","—","—","—","—",100]
jewishdf=jewishdf.drop(jewishdf.index[-1])
jewishdf["Jewish"]= pd.to_numeric(jewishdf["Jewish"],errors='coerce')
print(jewishdf["Jewish"].mean())
print(jewishdf["Jewish"].median())
print(jewishdf["Jewish"].max())
print(jewishdf.mode()["Jewish"][0])
print(jewishdf["Jewish"].min())
print(jewishdf["Jewish"].var())
print(jewishdf["Jewish"].isna().sum())
dflist.append(jewishdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_infant_and_under-five_mortality_rates"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
infantdeathdf=pd.read_html(string)
infantdeathdf=pd.DataFrame(infantdeathdf[0])
infantdeathdf.columns = ["Country", "InfantDeath"]
infantdeathdf["InfantDeath"]= pd.to_numeric(infantdeathdf["InfantDeath"],errors='coerce')
print(infantdeathdf["InfantDeath"].mean())
print(infantdeathdf["InfantDeath"].median())
print(infantdeathdf["InfantDeath"].max())
print(infantdeathdf.mode()["InfantDeath"][0])
print(infantdeathdf["InfantDeath"].min())
print(infantdeathdf["InfantDeath"].var())
print(infantdeathdf["InfantDeath"].isna().sum())
dflist.append(infantdeathdf)


wikiurl="https://en.wikipedia.org/wiki/Age_of_criminal_responsibility"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
criminalagedf=pd.read_html(string)
criminalagedf=pd.DataFrame(criminalagedf[0])
criminalagedf = criminalagedf.drop(criminalagedf.columns[[2,3,4,5]], axis=1)
criminalagedf.columns = ["Country", "CriminalAge"]
criminalagedf['CriminalAge'] = criminalagedf['CriminalAge'].str.replace('\[(.*?)\]', '', regex=True)
criminalagedf["CriminalAge"]= pd.to_numeric(criminalagedf["CriminalAge"],errors='coerce')
print(criminalagedf["CriminalAge"].mean())
print(criminalagedf["CriminalAge"].median())
print(criminalagedf["CriminalAge"].max())
print(criminalagedf.mode()["CriminalAge"][0])
print(criminalagedf["CriminalAge"].min())
print(criminalagedf["CriminalAge"].var())
print(criminalagedf["CriminalAge"].isna().sum())
dflist.append(criminalagedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_minimum_wage"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
minimumwage=pd.read_html(string)
minimumwage=pd.DataFrame(minimumwage[0])
minimumwage = minimumwage.drop(minimumwage.columns[[1,3,4,5,6,7,8]], axis=1)
minimumwage.columns = ["Country", "MinimumWage"]
minimumwage["MinimumWage"]= pd.to_numeric(minimumwage["MinimumWage"],errors='coerce')
print(minimumwage["MinimumWage"].mean())
print(minimumwage["MinimumWage"].median())
print(minimumwage["MinimumWage"].max())
print(minimumwage.mode()["MinimumWage"][0])
print(minimumwage["MinimumWage"].min())
print(minimumwage["MinimumWage"].var())
print(minimumwage["MinimumWage"].isna().sum())
dflist.append(minimumwage)



wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_external_debt"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
debtdf=pd.read_html(string)
debtdf=pd.DataFrame(debtdf[0])
debtdf = debtdf.drop(debtdf.columns[[1,2,3]], axis=1)
debtdf.columns = ["Country", "Debt"]
debtdf["Debt"]= pd.to_numeric(debtdf["Debt"],errors='coerce')
print(debtdf["Debt"].mean())
print(debtdf["Debt"].median())
print(debtdf["Debt"].max())
print(debtdf.mode()["Debt"][0])
print(debtdf["Debt"].min())
print(debtdf["Debt"].var())
print(debtdf["Debt"].isna().sum())
dflist.append(debtdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_income_equality"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
incomeinequalitydf=pd.read_html(string)
incomeinequalitydf=pd.DataFrame(incomeinequalitydf[0])
incomeinequalitydf = incomeinequalitydf.drop(incomeinequalitydf.columns[[1,2,3,4,5,6,8]], axis=1)
incomeinequalitydf.columns = ["Country", "IncomeInequality"]
incomeinequalitydf["IncomeInequality"]= pd.to_numeric(incomeinequalitydf["IncomeInequality"],errors='coerce')
incomeinequalitydf.drop(incomeinequalitydf.index[0],inplace=True)
print(incomeinequalitydf["IncomeInequality"].mean())
print(incomeinequalitydf["IncomeInequality"].median())
print(incomeinequalitydf["IncomeInequality"].max())
print(incomeinequalitydf.mode()["IncomeInequality"][0])
print(incomeinequalitydf["IncomeInequality"].min())
print(incomeinequalitydf["IncomeInequality"].var())
print(incomeinequalitydf["IncomeInequality"].isna().sum())
dflist.append(incomeinequalitydf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_total_health_expenditure_per_capita"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
HealthCostdf=pd.read_html(string)
HealthCostdf=pd.DataFrame(HealthCostdf[1])
HealthCostdf = HealthCostdf.drop(HealthCostdf.columns[[2,3,4]], axis=1)
HealthCostdf.columns = ["Country", "HealthExpenditure"]
HealthCostdf["HealthExpenditure"]= pd.to_numeric(HealthCostdf["HealthExpenditure"],errors='coerce')
print(HealthCostdf["HealthExpenditure"].mean())
print(HealthCostdf["HealthExpenditure"].median())
print(HealthCostdf["HealthExpenditure"].max())
print(HealthCostdf.mode()["HealthExpenditure"][0])
print(HealthCostdf["HealthExpenditure"].min())
print(HealthCostdf["HealthExpenditure"].var())
print(HealthCostdf["HealthExpenditure"].isna().sum())
dflist.append(HealthCostdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_suicide_rate"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
suicidedf=pd.read_html(string)
suicidedf=pd.DataFrame(suicidedf[0])
suicidedf = suicidedf.drop(suicidedf.columns[[2,3]], axis=1)
suicidedf.columns = ["Country", "Suicide"]
suicidedf["Suicide"]= pd.to_numeric(suicidedf["Suicide"],errors='coerce')
print(suicidedf["Suicide"].mean())
print(suicidedf["Suicide"].median())
print(suicidedf["Suicide"].max())
print(suicidedf.mode()["Suicide"][0])
print(suicidedf["Suicide"].min())
print(suicidedf["Suicide"].var())
print(suicidedf["Suicide"].isna().sum())
dflist.append(suicidedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_sovereign_states_and_dependencies_by_total_fertility_rate"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"}, limit=2)
string = str(table)
TFRdf=pd.read_html(string)
TFRdf=pd.DataFrame(TFRdf[1])
TFRdf = TFRdf.drop(TFRdf.columns[[0]], axis=1)
TFRdf.columns = ["Country", "Fertility"]
TFRdf['Fertility'] = TFRdf['Fertility'].str.replace('\((.*?)\)', '',regex=True)
TFRdf["Fertility"]= pd.to_numeric(TFRdf["Fertility"],errors='coerce')
print(TFRdf["Fertility"].mean())
print(TFRdf["Fertility"].median())
print(TFRdf["Fertility"].max())
print(TFRdf.mode()["Fertility"][0])
print(TFRdf["Fertility"].min())
print(TFRdf["Fertility"].var())
print(TFRdf["Fertility"].isna().sum())
dflist.append(TFRdf)


wikiurl="https://en.wikipedia.org/wiki/Tobacco_consumption_by_country"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
tobaccodf=pd.read_html(string)
tobaccodf=pd.DataFrame(tobaccodf[0])
tobaccodf.columns = ["Country", "Tobacco"]
tobaccodf["Tobacco"]= pd.to_numeric(tobaccodf["Tobacco"],errors='coerce')
print(tobaccodf["Tobacco"].mean())
print(tobaccodf["Tobacco"].median())
print(tobaccodf["Tobacco"].max())
print(tobaccodf.mode()["Tobacco"][0])
print(tobaccodf["Tobacco"].min())
print(tobaccodf["Tobacco"].var())
print(tobaccodf["Tobacco"].isna().sum())
dflist.append(tobaccodf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_obesity_rate"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
obesitydf=pd.read_html(string)
obesitydf=pd.DataFrame(obesitydf[0])
obesitydf.columns = ["Country", "Obesity"]
obesitydf["Obesity"]= pd.to_numeric(obesitydf["Obesity"],errors='coerce')
print(obesitydf["Obesity"].mean())
print(obesitydf["Obesity"].median())
print(obesitydf["Obesity"].max())
print(obesitydf.mode()["Obesity"][0])
print(obesitydf["Obesity"].min())
print(obesitydf["Obesity"].var())
print(obesitydf["Obesity"].isna().sum())
dflist.append(obesitydf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_number_of_Internet_users"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
internetusersdf=pd.read_html(string)
internetusersdf=pd.DataFrame(internetusersdf[2])
internetusersdf = internetusersdf.drop(internetusersdf.columns[[1,2,4,5,6,7]], axis=1)
internetusersdf.columns = ["Country", "InternetUsers"]
internetusersdf["InternetUsers"] = [72.40,59.50,91.20,71.60,77.10,63.80,60.80,89.50,75.90,94.20,71.30,88.80,93.30,64.10,82.50,69.90,96.60,92.20,50.10,95.40,89.30,75.50,85.30,90.60,89.00,74.10,53.60,59.30,75.20,75.50,71.40,59.60,59.50,91.90,81.60,72.90,16.20,43.40,49.00,30.40,90.70,46.50,76.30,72.50,64.90,27.40,45.90,38.80,22.20,86.30,91.30,53.50,47.20,95.20,16.70,79.50,91.40,75.90,75.30,86.10,74.10,22.90,76.80,32.70,73.60,7.30,62.90,66.70,76.30,87.90,58.10,52.20,53.90,29.30,84.70,19.20,50.10,20.50,92.40,32.80,94.80,40.10,87.30,81.10,24.40,85.00,27.80,65.20,28.00,81.60,83.30,12.40,62.10,10.10,95.40,80.70,82.20,68.10,62.50,100.70,13.80,29.00,69.20,81.80,19.70,12.90,94.20,8.70,70.90,54.50,91.70,68.90,10.80,35.40,80.50,8.70,73.70,73.20,31.60,20.10,23.60,25.30,77.40,75.60,84.60,12.10,10.70,97.80,49.90,20.60,11.80,19.30,86.80,69.40,44.40,9.30,11.90,5.60,11.10,76.50,36.70,36.90,9.30,19.90,21.80,54.10,29.20,4.80,89.30,48.20,75.40,7.80,48.90,80.40,15.80,34.70,91.30,50.00,7.30,27.00,65.50,82.40,20.40,88.90,53.10,1.70,36.10,45.00,52.90,83.10,78.70,67.70,3.70,44.10,77.50,96.00,50.60,83.20,96.30,3.50,10.30,69.10,22.30,8.40,67.50,1.80,30.20,51.10,27.40,94.00,52.30,71.10,73.20,90.90,93.80,42.00,69.80,35.50,102.40,7.00,95.30,99.50,48.90,59.60,13.20,46.40,76.40,51.80,49.30,53.80,76.50,64.10,11.90,53.40]
internetusersdf["InternetUsers"]= pd.to_numeric(internetusersdf["InternetUsers"],errors='coerce')
print(internetusersdf["InternetUsers"].mean())
print(internetusersdf["InternetUsers"].median())
print(internetusersdf["InternetUsers"].max())
print(internetusersdf.mode()["InternetUsers"][0])
print(internetusersdf["InternetUsers"].min())
print(internetusersdf["InternetUsers"].var())
print(internetusersdf["InternetUsers"].isna().sum())
dflist.append(internetusersdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_median_age"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
agedf=pd.read_html(string)
agedf=pd.DataFrame(agedf[0])
agedf = agedf.drop(agedf.columns[[1,2,4,5,6]], axis=1)
agedf.columns = ["Country", "MedianAge"]
agedf["MedianAge"]= pd.to_numeric(agedf["MedianAge"],errors='coerce')
print(agedf["MedianAge"].mean())
print(agedf["MedianAge"].median())
print(agedf["MedianAge"].max())
print(agedf.mode()["MedianAge"][0])
print(agedf["MedianAge"].min())
print(agedf["MedianAge"].var())
print(agedf["MedianAge"].isna().sum())
dflist.append(agedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_economic_freedom"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
economicfreedomdf=pd.read_html(string)
economicfreedomdf=pd.DataFrame(economicfreedomdf[0])
economicfreedomdf = economicfreedomdf.drop(economicfreedomdf.columns[[0,3]], axis=1)
economicfreedomdf.columns = ["Country", "EconomicFreedom"]
economicfreedomdf["EconomicFreedom"]= pd.to_numeric(economicfreedomdf["EconomicFreedom"],errors='coerce')
print(economicfreedomdf["EconomicFreedom"].mean())
print(economicfreedomdf["EconomicFreedom"].median())
print(economicfreedomdf["EconomicFreedom"].max())
print(economicfreedomdf.mode()["EconomicFreedom"][0])
print(economicfreedomdf["EconomicFreedom"].min())
print(economicfreedomdf["EconomicFreedom"].var())
print(economicfreedomdf["EconomicFreedom"].isna().sum())
dflist.append(economicfreedomdf) 


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_oil_production"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
oildf=pd.read_html(string)
oildf=pd.DataFrame(oildf[0])
oildf = oildf.drop(oildf.columns[[1]], axis=1)
oildf.columns = ["Country", "OilProduction"]
oildf["OilProduction"]= pd.to_numeric(oildf["OilProduction"],errors='coerce')
print(oildf["OilProduction"].mean())
print(oildf["OilProduction"].median())
print(oildf["OilProduction"].max())
print(oildf.mode()["OilProduction"][0])
print(oildf["OilProduction"].min())
print(oildf["OilProduction"].var())
print(oildf["OilProduction"].isna().sum())
dflist.append(oildf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_population_growth_rate"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
populationgrowth=pd.read_html(string)
populationgrowth=pd.DataFrame(populationgrowth[0])
populationgrowth = populationgrowth.drop(populationgrowth.columns[[1,2,3,4,5]], axis=1)
populationgrowth.columns = ["Country", "PopulationGrowth"]
populationgrowth["PopulationGrowth"]= pd.to_numeric(populationgrowth["PopulationGrowth"],errors='coerce')
print(populationgrowth["PopulationGrowth"].mean())
print(populationgrowth["PopulationGrowth"].median())
print(populationgrowth["PopulationGrowth"].max())
print(populationgrowth.mode()["PopulationGrowth"][0])
print(populationgrowth["PopulationGrowth"].min())
print(populationgrowth["PopulationGrowth"].var())
print(populationgrowth["PopulationGrowth"].isna().sum())
dflist.append(populationgrowth)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_life_expectancy"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
lifeexpectancy=pd.read_html(string)
lifeexpectancy=pd.DataFrame(lifeexpectancy[2])
lifeexpectancy = lifeexpectancy.drop(lifeexpectancy.columns[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]], axis=1)
lifeexpectancy.columns = ["Country", "LifeExpectancy"]
lifeexpectancy["LifeExpectancy"]= pd.to_numeric(lifeexpectancy["LifeExpectancy"],errors='coerce')
print(lifeexpectancy["LifeExpectancy"].mean())
print(lifeexpectancy["LifeExpectancy"].median())
print(lifeexpectancy["LifeExpectancy"].max())
print(lifeexpectancy.mode()["LifeExpectancy"][0])
print(lifeexpectancy["LifeExpectancy"].min())
print(lifeexpectancy["LifeExpectancy"].var())
print(lifeexpectancy["LifeExpectancy"].isna().sum())
dflist.append(lifeexpectancy)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_meat_consumption"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
meatconsumptiondf=pd.read_html(string)
meatconsumptiondf=pd.DataFrame(meatconsumptiondf[0])
meatconsumptiondf = meatconsumptiondf.drop(meatconsumptiondf.columns[[2,3,4,5,6]], axis=1)
meatconsumptiondf.columns = ["Country", "MeatConsumption"]
meatconsumptiondf["MeatConsumption"]= pd.to_numeric(meatconsumptiondf["MeatConsumption"],errors='coerce')
print(meatconsumptiondf["MeatConsumption"].mean())
print(meatconsumptiondf["MeatConsumption"].median())
print(meatconsumptiondf["MeatConsumption"].max())
print(meatconsumptiondf.mode()["MeatConsumption"][0])
print(meatconsumptiondf["MeatConsumption"].min())
print(meatconsumptiondf["MeatConsumption"].var())
print(meatconsumptiondf["MeatConsumption"].isna().sum())
dflist.append(meatconsumptiondf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_incarceration_rate"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
IncarcerationRatedf=pd.read_html(string)
IncarcerationRatedf=pd.DataFrame(IncarcerationRatedf[0])
IncarcerationRatedf = IncarcerationRatedf.drop(IncarcerationRatedf.columns[[1,2,4,5,6,7,8,9]], axis=1)
IncarcerationRatedf.columns = ["Country", "IncarcerationRate"]
IncarcerationRatedf["IncarcerationRate"]= pd.to_numeric(IncarcerationRatedf["IncarcerationRate"],errors='coerce')
print(IncarcerationRatedf["IncarcerationRate"].mean())
print(IncarcerationRatedf["IncarcerationRate"].median())
print(IncarcerationRatedf["IncarcerationRate"].max())
print(IncarcerationRatedf.mode()["IncarcerationRate"][0])
print(IncarcerationRatedf["IncarcerationRate"].min())
print(IncarcerationRatedf["IncarcerationRate"].var())
print(IncarcerationRatedf["IncarcerationRate"].isna().sum())
dflist.append(IncarcerationRatedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_literacy_rate"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
literacyratedf=pd.read_html(string)
literacyratedf=pd.DataFrame(literacyratedf[0])
literacyratedf = literacyratedf.drop(literacyratedf.columns[[1,2,3,4,6,7,8]], axis=1)
literacyratedf.columns = ["Country", "LiteracyRate"]
literacyratedf['LiteracyRate'] = literacyratedf['LiteracyRate'].str.replace('\[(.*)?\]', '',regex=True)
literacyratedf["LiteracyRate"]= pd.to_numeric(literacyratedf["LiteracyRate"],errors='coerce')
print(literacyratedf["LiteracyRate"].mean())
print(literacyratedf["LiteracyRate"].median())
print(literacyratedf["LiteracyRate"].max())
print(literacyratedf.mode()["LiteracyRate"][0])
print(literacyratedf["LiteracyRate"].min())
print(literacyratedf["LiteracyRate"].var())
print(literacyratedf["LiteracyRate"].isna().sum())
dflist.append(literacyratedf) 


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_age_at_first_marriage"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
firstmarriagedf=pd.read_html(string)
firstmarriagedf1=pd.DataFrame(firstmarriagedf[0])
firstmarriagedf2=pd.DataFrame(firstmarriagedf[1])
firstmarriagedf3=pd.DataFrame(firstmarriagedf[2])
firstmarriagedf4=pd.DataFrame(firstmarriagedf[3])
firstmarriagedf5=pd.DataFrame(firstmarriagedf[4])
frames = [firstmarriagedf1, firstmarriagedf2, firstmarriagedf3,firstmarriagedf4,firstmarriagedf5]
firstmarriagedf = pd.concat(frames)
firstmarriagedf = firstmarriagedf.drop(firstmarriagedf.columns[[1,3,4,5,6,7,8]], axis=1)
firstmarriagedf.columns = ["Country", "FirstMarriage"]
firstmarriagedf["FirstMarriage"]= pd.to_numeric(firstmarriagedf["FirstMarriage"],errors='coerce')
print(firstmarriagedf["FirstMarriage"].mean())
print(firstmarriagedf["FirstMarriage"].median())
print(firstmarriagedf["FirstMarriage"].max())
print(firstmarriagedf.mode()["FirstMarriage"][0])
print(firstmarriagedf["FirstMarriage"].min())
print(firstmarriagedf["FirstMarriage"].var())
print(firstmarriagedf["FirstMarriage"].isna().sum())
dflist.append(firstmarriagedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_spending_on_education_(%25_of_GDP)"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
educationspendingdf=pd.read_html(string)
educationspendingdf=pd.DataFrame(educationspendingdf[0])
educationspendingdf = educationspendingdf.drop(educationspendingdf.columns[[2,3]], axis=1)
educationspendingdf.columns = ["Country", "EducationSpending"]
educationspendingdf["EducationSpending"]= pd.to_numeric(educationspendingdf["EducationSpending"],errors='coerce')
print(educationspendingdf["EducationSpending"].mean())
print(educationspendingdf["EducationSpending"].median())
print(educationspendingdf["EducationSpending"].max())
print(educationspendingdf.mode()["EducationSpending"][0])
print(educationspendingdf["EducationSpending"].min())
print(educationspendingdf["EducationSpending"].var())
print(educationspendingdf["EducationSpending"].isna().sum())
dflist.append(educationspendingdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_homeless_population"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
homelesspopulationdf=pd.read_html(string)
homelesspopulationdf=pd.DataFrame(homelesspopulationdf[0])
homelesspopulationdf = homelesspopulationdf.drop(homelesspopulationdf.columns[[1,2,4,5,6]], axis=1)
homelesspopulationdf.columns = ["Country", "HomelessPopulation"]
homelesspopulationdf["HomelessPopulation"]= pd.to_numeric(homelesspopulationdf["HomelessPopulation"],errors='coerce')
print(homelesspopulationdf["HomelessPopulation"].mean())
print(homelesspopulationdf["HomelessPopulation"].median())
print(homelesspopulationdf["HomelessPopulation"].max())
print(homelesspopulationdf.mode()["HomelessPopulation"][0])
print(homelesspopulationdf["HomelessPopulation"].min())
print(homelesspopulationdf["HomelessPopulation"].var())
print(homelesspopulationdf["HomelessPopulation"].isna().sum())
dflist.append(homelesspopulationdf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_milk_consumption_per_capita"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
milkconsumptiondf=pd.read_html(string)
milkconsumptiondf=pd.DataFrame(milkconsumptiondf[0])
milkconsumptiondf = milkconsumptiondf.drop(milkconsumptiondf.columns[[0,1,4]], axis=1)
milkconsumptiondf.columns = ["Country", "MilkConsumption"]
milkconsumptiondf["MilkConsumption"]= pd.to_numeric(milkconsumptiondf["MilkConsumption"],errors='coerce')
print(milkconsumptiondf["MilkConsumption"].mean())
print(milkconsumptiondf["MilkConsumption"].median())
print(milkconsumptiondf["MilkConsumption"].max())
print(milkconsumptiondf.mode()["MilkConsumption"][0])
print(milkconsumptiondf["MilkConsumption"].min())
print(milkconsumptiondf["MilkConsumption"].var())
print(milkconsumptiondf["MilkConsumption"].isna().sum())
dflist.append(milkconsumptiondf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_number_of_scientific_and_technical_journal_articles"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
articlesdf=pd.read_html(string)
articlesdf=pd.DataFrame(articlesdf[0])
articlesdf = articlesdf.drop(articlesdf.columns[[0,2]], axis=1)
articlesdf.columns = ["Country", "ScientificPublications"]
articlesdf["ScientificPublications"]= pd.to_numeric(articlesdf["ScientificPublications"],errors='coerce')
print(articlesdf["ScientificPublications"].mean())
print(articlesdf["ScientificPublications"].median())
print(articlesdf["ScientificPublications"].max())
print(articlesdf.mode()["ScientificPublications"][0])
print(articlesdf["ScientificPublications"].min())
print(articlesdf["ScientificPublications"].var())
print(articlesdf["ScientificPublications"].isna().sum())
dflist.append(articlesdf)


wikiurl="https://en.wikipedia.org/wiki/Books_published_per_country_per_year"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
booksdf=pd.read_html(string)
booksdf=pd.DataFrame(booksdf[0])
booksdf = booksdf.drop(booksdf.columns[[0,2,4,5]], axis=1)
booksdf.columns = ["Country", "BooksPublished"]
booksdf['BooksPublished'] = booksdf['BooksPublished'].str.replace('\+', '',regex=True)
booksdf["BooksPublished"]= pd.to_numeric(booksdf["BooksPublished"],errors='coerce')
print(booksdf["BooksPublished"].mean())
print(booksdf["BooksPublished"].median())
print(booksdf["BooksPublished"].max())
print(booksdf.mode()["BooksPublished"][0])
print(booksdf["BooksPublished"].min())
print(booksdf["BooksPublished"].var())
print(booksdf["BooksPublished"].isna().sum())
dflist.append(booksdf) 


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_food_energy_intake"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
energyintakedf=pd.read_html(string)
energyintakedf=pd.DataFrame(energyintakedf[0])
energyintakedf = energyintakedf.drop(energyintakedf.columns[[0,3]], axis=1)
energyintakedf.columns = ["Country", "EnergyIntake"]
energyintakedf["EnergyIntake"]= pd.to_numeric(energyintakedf["EnergyIntake"],errors='coerce')
print(energyintakedf["EnergyIntake"].mean())
print(energyintakedf["EnergyIntake"].median())
print(energyintakedf["EnergyIntake"].max())
print(energyintakedf.mode()["EnergyIntake"][0])
print(energyintakedf["EnergyIntake"].min())
print(energyintakedf["EnergyIntake"].var())
print(energyintakedf["EnergyIntake"].isna().sum())
dflist.append(energyintakedf)


wikiurl="https://en.wikipedia.org/wiki/List_of_countries_by_average_yearly_temperature"
response=requests.get(wikiurl)
soup = BeautifulSoup(response.text, 'html.parser')
table=soup.find_all('table',{'class':"wikitable"})
string = str(table)
averagetempdf=pd.read_html(string)
averagetempdf=pd.DataFrame(averagetempdf[0])
averagetempdf.columns = ["Country", "AverageTemperature"]
averagetempdf["AverageTemperature"]= pd.to_numeric(averagetempdf["AverageTemperature"],errors='coerce')
print(averagetempdf["AverageTemperature"].mean())
print(averagetempdf["AverageTemperature"].median())
print(averagetempdf["AverageTemperature"].max())
print(averagetempdf.mode()["AverageTemperature"][0])
print(averagetempdf["AverageTemperature"].min())
print(averagetempdf["AverageTemperature"].var())
print(averagetempdf["AverageTemperature"].isna().sum())
dflist.append(averagetempdf)


countrylist = []
for df in dflist:
    originaldfcountrylist = []
    newdfcountrylist = []
    for country in df["Country"]:
        originaldfcountrylist.append(country)
        country = str(country)
        country = re.sub("\\u202f\*", "", country)
        country = re.sub("\[(.*?)\]", "", country)
        country = re.sub('\((.*?)\)', "", country)
        country = country.strip()
        newdfcountrylist.append(country)
    
        if country not in countrylist:
            countrylist.append(country)
    df["Country"].replace(originaldfcountrylist,newdfcountrylist, inplace=True)   



originaldf = pd.DataFrame(countrylist, columns=["Country"])
for df in dflist:
    originaldf = originaldf.set_index('Country').join(df.set_index('Country')).reset_index()

originaldf = originaldf.dropna(thresh=28)


mediandf = pd.DataFrame(originaldf["Country"])
for column in originaldf.columns:
    if column != "Country":
        mediandf[column] = originaldf[column].fillna(originaldf[column].median())
    if column == "OilProduction":
        mediandf[column] = originaldf[column].fillna(0)



model = linear_model.LinearRegression()
regression1df = pd.DataFrame(originaldf["Country"])
for column in mediandf.columns:

    if column != "Country":
      model.fit(mediandf.drop(["Country",column], axis=1),mediandf[column])
      regression1df[column] = model.predict(mediandf.drop(["Country",column], axis=1))
      

regression1df = originaldf.fillna(regression1df)

regression2df = pd.DataFrame(originaldf["Country"])
for column in regression1df.columns:

    if column != "Country":
      model.fit(regression1df.drop(["Country",column], axis=1),regression1df[column])
      regression2df[column] = model.predict(regression1df.drop(["Country",column], axis=1))
      

regression2df = originaldf.fillna(regression2df)

zeroonedf = pd.DataFrame(originaldf["Country"])
for column in regression2df.columns:
  if column != "Country":
    zeroonedf[column] = regression2df[column].map(lambda x: 0 if x <= regression2df[column].median() else 1)


#2)
corrmatrix = regression2df.corr(numeric_only=True)
strongestcorrint = []
strongestcorrname = []

for column in corrmatrix:
  strongestcorrname.append([corrmatrix[column].abs().nlargest(2).index[0],corrmatrix[column].abs().nlargest(2).index[1]])
  strongestcorrint.append(corrmatrix.loc[corrmatrix[column].abs().nlargest(2).index[0],corrmatrix[column].abs().nlargest(2).index[1]])

print(strongestcorrint)
print(strongestcorrname)


meandf = pd.DataFrame()
for column in corrmatrix:
  mean = corrmatrix[column].abs().mean()
  meandf[column] = [mean]

meandf =meandf.sort_values(by=0,axis=1,ascending=False)
print(meandf.columns)
print(meandf.values.tolist())



#3)
#a)
regression2dflr =  regression2df.drop(['Country'], axis = 1)
for target in regression2dflr:
    mse_lr = []
    X_train, X_test, y_train, y_test = train_test_split(regression2dflr.drop(target, axis = 1), regression2df[target], test_size=0.2, shuffle=False)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
        
    mse_lr.append(mean_squared_error(y_test, y_pred_lr))
    print(f'Linear Regression for {target}:')

    print('MSE:', mse_lr)


zeroOnedfnb = zeroonedf.drop(['Country'], axis = 1)
listde2Colonnennb = {}
for target in zeroOnedfnb:
    mse_nb = []
    X_train, X_test, y_train, y_test = train_test_split(zeroOnedfnb.drop([target], axis=1), zeroonedf[target], test_size=0.2)
        
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    
    mse_nb.append(mean_squared_error(y_test, y_pred_nb))
    print(f'Bayesian Classifier for {target}: ')
    
    print('MSE:', mse_nb)

#b)
allbestcolumns = []
for target in regression2dflr:
  min_mse = 10000000000000000
  bestcolumns = None
  for column1 in regression2dflr:
        if column1 == target:
            continue
        for column2 in regression2dflr:
          if column2 == target:
            continue
          if column2 == column1:
            continue
          X_train, X_test, y_train, y_test = train_test_split(regression2dflr[[column1,column2]], regression2df[target], test_size=0.2)
          lr = LinearRegression()
          lr.fit(X_train, y_train)
          y_pred_lr = lr.predict(X_test)
          current_mse= mean_squared_error(y_test, y_pred_lr)

          if min_mse > current_mse:
            min_mse = current_mse
            bestcolumns = [column1,column2]
  allbestcolumns.append([target,bestcolumns])
print(allbestcolumns)


allbestcolumns = []
for target in zeroOnedfnb:
  min_mse = 10000000000000000
  bestcolumns = None
  for column1 in zeroOnedfnb:
        if column1 == target:
            continue
        for column2 in zeroOnedfnb:
          if column2 == target:
            continue
          if column2 == column1:
            continue
          X_train, X_test, y_train, y_test = train_test_split(zeroOnedfnb[[column1,column2]], zeroonedf[target], test_size=0.2)
          nb = GaussianNB()
          nb.fit(X_train, y_train)
          y_pred_nb = nb.predict(X_test)
          current_mse= mean_squared_error(y_test, y_pred_nb)

          if min_mse > current_mse:
            min_mse = current_mse
            bestcolumns = [column1,column2]
  allbestcolumns.append([target,bestcolumns])
print(allbestcolumns)

#c)

for column in regression2dflr.columns:
    regression2dflr[column] = (regression2dflr[column] -regression2dflr[column].mean()) / regression2dflr[column].std()

pairmeanmses = []
for column1 in regression2dflr:
  for column2 in regression2dflr:
    pairmses = []
    if column2 == column1:
      continue
    for target in regression2dflr:
      if column1 == target:
        continue
      if column2 == target:
        continue
      X_train, X_test, y_train, y_test = train_test_split(regression2dflr[[column1,column2]], regression2df[target], test_size=0.2)
      lr = LinearRegression()
      lr.fit(X_train, y_train)
      y_pred_lr = lr.predict(X_test)
      current_mse= mean_squared_error(y_test, y_pred_lr)
      pairmses.append(current_mse)
    pairmeanmses.append([column1,column2,np.mean(pairmses)])
print(pairmeanmses)
print(min(pairmeanmses,key=itemgetter(2)))


pairmeanmses = []
for column1 in zeroOnedfnb:
  for column2 in zeroOnedfnb:
    pairmses = []
    if column2 == column1:
      continue
    for target in zeroOnedfnb:
      if column1 == target:
        continue
      if column2 == target:
        continue
      X_train, X_test, y_train, y_test = train_test_split(zeroOnedfnb[[column1,column2]], zeroonedf[target], test_size=0.2)
      lr = LinearRegression()
      lr.fit(X_train, y_train)
      y_pred_lr = lr.predict(X_test)
      current_mse= mean_squared_error(y_test, y_pred_lr)
      pairmses.append(current_mse)
    pairmeanmses.append([column1,column2,np.mean(pairmses)])
print(pairmeanmses)
print(min(pairmeanmses,key=itemgetter(2)))


#4)
regression2df = regression2df.reset_index(drop=True)

pcadf = pd.DataFrame(regression2df["Country"])
for column in regression2df.columns:
  if column != "Country":
    pcadf[column] = (regression2df[column] -regression2df[column].mean()) / regression2df[column].std()

pca = decomposition.PCA(2)
pca = pca.fit(pcadf.drop("Country", axis=1))
pcaX = pca.transform(pcadf.drop("Country", axis=1))
df = pd.DataFrame(pcaX)

df["Country"] = regression2df["Country"]
fig,ax = plt.subplots()
scatter = ax.scatter(df[0], df[1])
for i, txt in enumerate(df["Country"]):
  ax.annotate(txt,(df[0][i],df[1][i]), size=2)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("PCA à 2 dimensions")
plt.savefig('filename.png', dpi=500)


pca5 = decomposition.PCA(5)
pca5 = pca5.fit(pcadf.drop("Country", axis=1))
pcaX5 = pca5.transform(pcadf.drop("Country", axis=1))
df = pd.DataFrame(pcaX5)
df["Country"] = regression2df["Country"]

fig,ax = plt.subplots()
scatter = ax.scatter(df[2], df[3])
for i, txt in enumerate(df["Country"]):
  ax.annotate(txt,(df[2][i],df[3][i]), size=2)

plt.xlabel("Dimension 3")
plt.ylabel("Dimension 4")
plt.title("PCA à 5 dimensions")
plt.savefig('filename3.png', dpi=500)

fig,ax = plt.subplots()
scatter = ax.scatter(df[4],  len(df[4]) * [1])
for i, txt in enumerate(df["Country"]):
  ax.annotate(txt,(df[4][i], 1), size=2)

plt.title("PCA à 5 dimensions, 5ième dimension")
plt.savefig('filename4.png', dpi=500)

print(pca5.explained_variance_ratio_)