import re
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime
from datetime import datetime
import unicodedata
from unidecode import unidecode
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#data about injuries from transfermarkt
#based on the work of kaggle user, link https://www.kaggle.com/code/eliesemmel/kernelde4fae5abe/notebook

headers = {'User-Agent': 
           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'}
league = ['GB1','FR1','L1','IT1','ES1']
league_page = "https://www.transfermarkt.com/jumplist/startseite/wettbewerb/"

def get_club_details(tr_tag):
    club = tr_tag.find_all('a')
    club_link = club[0]['href']
    club_name = club[0]['title']
    return tuple((club_link,club_name))

clubs_list = []
for league_id in league:
    page = requests.get(league_page + league_id,headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('tbody')[1]
    tr_container = tbody_container.find_all('tr')
    for tr_tag in tr_container :
        clubs_list.append(get_club_details(tr_tag))
print('All the club were uploaded')

def get_players_club(player):
    player_link = player['href']
    player_name = player['title']
    player_id = player_link.split("/",-1)[len(player_link.split("/",-1))-1]
    return tuple((player_id,player_link,player_name,club_name))


url_site = "https://www.transfermarkt.com"
player_list = []
for club_link,club_name in clubs_list:
    page = requests.get(url_site + club_link,headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('tbody')[1]
    players_details = tbody_container.find_all('a')
    index_finder = tbody_container.find_all('a',{"href":"#"})
    player_det = []
    ind = 0
    for i in range(len(players_details)):
        if players_details[i]==index_finder[ind]:
            player_det.append(players_details[i+1])
            ind = ind + 1
            if ind == len(index_finder):
                break
    for player in player_det :
        player_list.append(get_players_club(player))
print('All the players were uploaded')

#link = '/alejandro-cantero/profil/spieler/638235'
#page = requests.get(url_site + link,headers = headers)
#soup = bs(page.content, 'html.parser')
#table_container = soup.find_all('span',class_="info-table__content info-table__content--bold")

#table_container[2] #birth
#table_container[7] #position
#table_container[8] #foot
#table_container[5] #height
#table_container[6] #nationality

#table_container[0] #birth
#table_container[3] #height
#table_container[4] #nationality
#table_container[5] #position
#table_container[6] #foot

def retrieve_birth(table_container,ind=1):
    b = str(table_container[ind])
    result = re.search('datum/(.*)">', b)
    return(result.group(1))

def retrieve_height(table_container,ind=4):
    b = str(table_container[ind].get_text())
    b = b.replace(u'\xa0m', u'')
    b = b.replace(u',', u'.')
    return(float(b))

def retrieve_position(table_container,ind=6):
    b = table_container[ind].get_text()
    return(b.strip())

def retrieve_foot(table_container,ind=7):
    b = table_container[ind].get_text()
    return(b.strip())

def retrieve_nationality(nationality_container):
    b = nationality_container.get_text()
    return(b.strip())



def get_profil_detail(soup):
    table_container = soup.find_all('span',class_="info-table__content info-table__content--bold")
    nationality_container = soup.find_all('span',class_="dataValue")[2]
    birth = retrieve_birth(table_container)
    height = retrieve_height(table_container)
    country = retrieve_nationality(nationality_container)
    role = retrieve_position(table_container)
    foot = retrieve_foot(table_container)
    tbody_container = soup.find_all('tbody')[0]
    tr_transfer_container = tbody_container.find_all('tr',class_="zeile-transfer")
    transfer_list = []
    #tr_transfer_tag=tr_transfer_container[0] #do usuniecia
    for tr_transfer_tag in tr_transfer_container:
        td_transfer_container = tr_transfer_tag.find_all("td")
        tranfer_from = td_transfer_container[5].find_all('img')[0]["alt"]
        transfer_to = td_transfer_container[2].find_all('img')[0]["alt"]
        transfer_season = td_transfer_container[0].get_text()
        transfer_date = td_transfer_container[1].get_text()
        transfer_list.append(tuple((tranfer_from,transfer_to,transfer_season,transfer_date)))
    return tuple((Id,name,club,birth,height,country,role,foot,link,transfer_list))

def get_weird_profil_detail(soup):
    table_container = soup.find_all('span',class_="info-table__content info-table__content--bold")
    nationality_container = soup.find_all('span',class_="dataValue")[2]
    birth = retrieve_birth(table_container,0)
    height = retrieve_height(table_container,3)
    country = retrieve_nationality(nationality_container)
    role = retrieve_position(table_container,5)
    foot = retrieve_foot(table_container,6)
    tbody_container = soup.find_all('tbody')[0]
    tr_transfer_container = tbody_container.find_all('tr',class_="zeile-transfer")
    transfer_list = []
    #tr_transfer_tag=tr_transfer_container[0] #do usuniecia
    for tr_transfer_tag in tr_transfer_container:
        td_transfer_container = tr_transfer_tag.find_all("td")
        tranfer_from = td_transfer_container[5].find_all('img')[0]["alt"]
        transfer_to = td_transfer_container[2].find_all('img')[0]["alt"]
        transfer_season = td_transfer_container[0].get_text()
        transfer_date = td_transfer_container[1].get_text()
        transfer_list.append(tuple((tranfer_from,transfer_to,transfer_season,transfer_date)))
    return tuple((Id,name,club,birth,height,country,role,foot,link,transfer_list))

def get_weirder_profil_detail(soup):
    table_container = soup.find_all('span',class_="info-table__content info-table__content--bold")
    nationality_container = soup.find_all('span',class_="dataValue")[2]
    birth = retrieve_birth(table_container,2)
    height = retrieve_height(table_container,5)
    country = retrieve_nationality(nationality_container)
    role = retrieve_position(table_container,7)
    foot = retrieve_foot(table_container,8)
    tbody_container = soup.find_all('tbody')[0]
    tr_transfer_container = tbody_container.find_all('tr',class_="zeile-transfer")
    transfer_list = []
    #tr_transfer_tag=tr_transfer_container[0] #do usuniecia
    for tr_transfer_tag in tr_transfer_container:
        td_transfer_container = tr_transfer_tag.find_all("td")
        tranfer_from = td_transfer_container[5].find_all('img')[0]["alt"]
        transfer_to = td_transfer_container[2].find_all('img')[0]["alt"]
        transfer_season = td_transfer_container[0].get_text()
        transfer_date = td_transfer_container[1].get_text()
        transfer_list.append(tuple((tranfer_from,transfer_to,transfer_season,transfer_date)))
    return tuple((Id,name,club,birth,height,country,role,foot,link,transfer_list))

player_details = []
i=0
for Id,link,name,club in player_list:
    i=i+1
    if i%500 == 0:
        print("new league upload")
    try:
        page = requests.get(url_site + link,headers = headers)
        soup = bs(page.content, 'html.parser')
        player_details.append(get_profil_detail(soup))
    except Exception as e1:
        try:
            page = requests.get(url_site + link,headers = headers)
            soup = bs(page.content, 'html.parser')
            player_details.append(get_weird_profil_detail(soup))
        except Exception as e2:
            try:
                page = requests.get(url_site + link,headers = headers)
                soup = bs(page.content, 'html.parser')
                player_details.append(get_weirder_profil_detail(soup))
            except Exception as e3:
                player_details.append(tuple((Id,name,club,None,None,None,None,None,link,[])))
                print(name)
                continue
print("all player details uploaded")

#soup.find_all('span',class_="info-table__content info-table__content--bold")[0] #name
#soup.find_all('span',class_="info-table__content info-table__content--bold")[1] #date of birth
#soup.find_all('span',class_="info-table__content info-table__content--bold")[2] #place of birth
#soup.find_all('span',class_="info-table__content info-table__content--bold")[3] #age
#soup.find_all('span',class_="info-table__content info-table__content--bold")[4] #heigth
#soup.find_all('span',class_="info-table__content info-table__content--bold")[5] #citizenship
#soup.find_all('span',class_="info-table__content info-table__content--bold")[6] #position
#soup.find_all('span',class_="info-table__content info-table__content--bold")[7] #foot


def get_injuries_details(link):
    injury_link = link.replace(u'profil', u'verletzungen')
    page = requests.get(url_site + injury_link,headers = headers)
    soup = bs(page.content, 'html.parser')
    tbody_container = soup.find_all('tbody')[0]
    tr_container = tbody_container.find_all('tr')
    injuries_list = []
    for tr_tag in tr_container:
        tr_tag_container = tr_tag.find_all('td')
        season = tr_tag_container[0]
        injury = tr_tag_container[1]
        date_from = tr_tag_container[2]
        date_to = tr_tag_container[3]
        time_out = tr_tag_container[4]
        injuries_list.append(tuple((season.get_text(),injury.get_text(),date_from.get_text(),date_to.get_text(),int(time_out.get_text().split()[0]))))
    return injuries_list

player_list = []
for Id,name,club,birth,height,country,role,foot,link,transfer_list in player_details:
    try:
        player_list.append(tuple((Id,name,club,birth,height,country,role,foot,transfer_list,get_injuries_details(link))))
    except Exception as e:
        player_list.append(tuple((Id,name,club,birth,height,country,role,foot,transfer_list,[])))
        continue
print("all player injuries details uploaded")
print("End of uploading from Transfermarkt")

#zapis danych, kontuzji i transferów do tabel

player_data = pd.DataFrame(columns=["Id","player","birth","height","nationality","position","foot"])
player_transfers = pd.DataFrame(columns=["Id","player","next_club","previous_club","transfer_date"])
player_injuries = pd.DataFrame(columns=["Id","player","injury_type","injury_from","injury_to","absence_days"])

for player in player_list:
    player_data = player_data.append({"Id":player[0],"player":player[1],"birth":player[3],"height":player[4],"nationality":player[5],"position":player[6],"foot":player[7]},ignore_index=True)
    for transfer in player[8]:
        player_transfers = player_transfers.append({"Id":player[0],"player":player[1],"next_club":transfer[0],"previous_club":transfer[1],"transfer_date":transfer[3]},ignore_index=True)
    for injury in player[9]:
        player_injuries = player_injuries.append({"Id":player[0],"player":player[1],"injury_type":injury[1],"injury_from":injury[2],"injury_to":injury[3],"absence_days":injury[4]},ignore_index=True)

pd.set_option('display.max_columns', None)
pd.set_option('max_row', None)
player_data.info()
player_data.head()
player_transfers.info()
player_injuries.info()
player_injuries.head(50)


player_data.to_csv("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Dane transfermarkt\\player_data.csv")
player_transfers.to_csv("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Dane transfermarkt\\player_transfers.csv")
player_injuries.to_csv("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Dane transfermarkt\\player_injuries.csv")

#column names in seasons

df = pd.read_excel("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Wyscout\\Premier League\\Arsenal\\Seasons\\Team Stats Arsenal.xlsx")
df.info()
df.head(10)
colnames = df.columns
season_columns = []
i = 0
while i < (len(colnames)-3):
    if colnames[i+1].startswith('Unnamed') and colnames[i+2].startswith('Unnamed') and colnames[i+3].startswith('Unnamed'):
        try:
            temp = colnames[i]
            season_columns.append(temp.split(" / ")[0])
            season_columns.append(temp.split(" / ")[0] + " " + temp.split(" / ")[1])
            season_columns.append(temp.split(" / ")[0] + " " + temp.split(" / ")[2])
            season_columns.append(temp.split(" / ")[0] + " " + temp.split(" / ")[3])
            i = i + 3
        except Exception as e:
            print(colnames[i])
    else:
        if colnames[i+1].startswith('Unnamed') and colnames[i+2].startswith('Unnamed'):
            try:
                temp = colnames[i]
                season_columns.append(temp.split(" / ")[0])
                season_columns.append(temp.split(" / ")[0] + " " + temp.split(" / ")[1])
                season_columns.append("Percentage of " + temp.split(" / ")[0] + " " + temp.split(" / ")[1])
                i = i + 2
            except Exception as e:
                print(colnames[i])
        else:
            if colnames[i].startswith('Unnamed'):
                i = i + 1
                continue
            else:
                season_columns.append(colnames[i])
                i = i + 1

while i < len(colnames):
    season_columns.append(colnames[i])
    i = i + 1

with open('C:\\Users\\Lenovo\\Desktop\\Magisterka\\season_columns.txt', 'w') as f:
    for item in season_columns:
        f.write("%s\n" % item)

df = pd.read_excel("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Wyscout\\Primera Division\\Athletic Bilbao\\Players\\Player stats Ander Capa.xlsx")
df.info()
df.head(10)
colnames = df.columns
players_columns = []
i = 0

while i < (len(colnames)-1):
            if colnames[i+1].startswith('Unnamed'):
                try:
                    temp = colnames[i]
                    players_columns.append(temp.split(" / ")[0])
                    players_columns.append(temp.split(" / ")[0] + " " + temp.split(" / ")[1])
                except Exception as e:
                    print(colnames[i])
                i = i + 2
            else:
                players_columns.append(colnames[i])
                i = i + 1

players_columns.append(colnames[i])

with open('C:\\Users\\Lenovo\\Desktop\\Magisterka\\players_columns.txt', 'w') as f:
    for item in players_columns:
        f.write("%s\n" % item)

