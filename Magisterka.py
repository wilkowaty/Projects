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
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize


def date_converter(x):
    try:
        d = datetime.strptime(x,"%Y-%m-%d")
    except Exception as e:
        try:
            d = datetime.strptime(x,"%b %d, %Y")
        except Exception as ee:
            d = None
    return d

clubs_dict = {
    'AC Milan':'Milan',
    'ACF Fiorentina':'Fiorentina',
    'AD Alcorcon':'Alcorcon',
    'AS Monaco':'Monaco',
    'AS Roma':'Roma',
    'Arsenal FC':'Arsenal',
    'Atalanta BC':'Atalanta',
    'Athletic Bilbao':'Athletic Bilbao',
    'Atletico de Madrid':'Atletico Madrid',
    'Bayer 04 Leverkusen':'Bayer Leverkusen',
    'Bayern Munich':'Bayern Munchen',
    'Borussia Dortmund':'Borussia Dortmund',
    'Brighton & Hove Albion':'Brighton',
    'CA Osasuna':'Osasuna',
    'CD Leganes':'Leganes',
    'CD Lugo':'Lugo',
    'CD Mirandes':'Mirandes',
    'CD Tenerife':'Tenerife',
    'CF Fuenlabrada':'Fuenlabrada',
    'CF Os Belenenses':'Belenenses',
    'Celta de Vigo':'Celta de Vigo',
    'Chelsea FC':'Chelsea',
    'Crystal Palace':'Crystal Palace',
    'Cadiz CF':'Cadiz',
    'Deportivo Alaves':'Deportivo Alaves',
    'Eintracht Frankfurt':'Eintracht Frankfurt',
    'Elche CF':'Elche',
    'Everton FC':'Everton',
    'FC Barcelona':'Barcelona',
    'FC Cartagena':'Cartagena',
    'FC Girondins Bordeaux':'Bordeaux',
    'FC Lorient':'Lorient',
    'FC Metz':'Metz',
    'FC Nantes':'Nantes',
    'FC Porto':'Porto',
    'Genoa CFC':'Genoa',
    'Getafe CF':'Getafe',
    'Girona FC':'Girona',
    'Granada CF':'Granada',
    'Hertha BSC':'Hertha BSC',
    'Inter Milan':'Internazionale',
    'Juventus FC':'Juventus',
    'LOSC Lille':'Lille',
    'Leeds United':'Leeds United',
    'Leicester City':'Leicester City',
    'Levante UD':'Levante',
    'Liverpool FC':'Liverpool',
    'Manchester City':'Manchester City',
    'Manchester United':'Manchester United',
    'Montpellier HSC':'Montpellier',
    'Moreirense FC':'Moreirense',
    'Malaga CF':'Malaga',
    'Newcastle United':'Newcastle United',
    'Norwich City':'Norwich City',
    'OGC Nice':'Nice',
    'Olympique Lyon':'Olympique Lyonnais',
    'Olympique Marseille':'Olympique Marseille',
    'Paris Saint-Germain':'PSG',
    'RB Leipzig':'RB Leipzig',
    'RC Strasbourg Alsace':'Strasbourg',
    'RCD Espanyol Barcelona':'Espanyol',
    'RCD Mallorca':'Mallorca',
    'Rayo Vallecano':'Rayo Vallecano',
    'Real Betis Balompie':'Real Betis',
    'Real Madrid':'Real Madrid',
    'Real Oviedo':'Real Oviedo',
    'Real Sociedad':'Real Sociedad',
    'Real Sociedad B':'Real Sociedad B',
    'Real Valladolid CF':'Real Valladoid',
    'Real Zaragoza':'Real Zaragoza',
    'SC Braga':'Sporting Braga',
    'SCO Angers':'Angers SCO',
    'SD Amorebieta':'Amorabieta',
    'SD Eibar':'Eibar',
    'SD Huesca':'Huesca',
    'SD Ponferradina':'Ponferradina',
    'SL Benfica':'Benfica',
    'SSC Napoli':'Napoli',
    'Sevilla FC':'Sevilla',
    'Southampton FC':'Southampton',
    'Spezia Calcio':'Spezia',
    'Sporting CP':'Sporting CP',
    'Sporting Gijon':'Sporting Gijon',
    'Stade Brest 29 B':'Brest',
    'Stade Reims':'Reims',
    'Stade Rennais FC':'Rennes',
    'Torino FC':'Torino',
    'Tottenham Hotspur':'Tottenham Hotspur',
    'UC Sampdoria':'Sampdoria',
    'UD Almeria':'Almeria',
    'UD Las Palmas':'Las Palmas',
    'US Sassuolo':'Sassuolo',
    'Udinese Calcio':'Udinese',
    'Valencia CF':'Valencia',
    'VfB Stuttgart':'Stuttgart',
    'VfL Wolfsburg':'Wolfsburg',
    'Villarreal CF':'Villareal',
    'Vitoria Guimaraes SC':'Vitoria Guimaraes',
    'Watford FC':'Watford',
    'West Ham United':'West Ham United',
    'Wolverhampton Wanderers':'Wolverhampton Wanderers'}

def club_name_converter(x):
    y = unidecode(x)
    if y in clubs_dict.keys():
        return clubs_dict[y]
    else:
        return 'other'

player_data = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Dane transfermarkt\\player_data.csv",index_col = 0)
player_transfers = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Dane transfermarkt\\player_transfers.csv",index_col = 0)
player_injuries = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\Magisterka\\Dane transfermarkt\\player_injuries.csv",index_col = 0)

player_data["birth"] = player_data["birth"].apply(date_converter)
player_data["player"] = player_data["player"].apply(unidecode)
player_transfers["transfer_date"] = player_transfers["transfer_date"].apply(date_converter)
player_transfers["player"] = player_transfers["player"].apply(unidecode)
player_transfers["next_club"] = player_transfers["next_club"].apply(club_name_converter)
player_transfers["previous_club"] = player_transfers["previous_club"].apply(club_name_converter)
player_injuries["injury_from"] = player_injuries["injury_from"].apply(date_converter)
player_injuries["injury_to"] = player_injuries["injury_to"].apply(date_converter)
player_injuries["player"] = player_injuries["player"].apply(unidecode)
previous_injuries = player_injuries.groupby("Id").Id.apply(lambda x:np.arange(start=0,stop=len(x),step=1))
#player_injuries["previous_injuries"] = [number for i in previous_injuries.index for number in previous_injuries[i] ]



# opening the file in read mode
my_file = open("C:\\Users\\Lenovo\\Desktop\\Magisterka\\season_columns.txt", "r")
  
# reading the file
data = my_file.read()
  
# replacing end splitting the text 
# when newline ('\n') is seen.
season_columns = data.split("\n")
my_file.close()
season_columns = [name + '_match' for name in season_columns]
#-------------------------------------------------

#column names in players


# opening the file in read mode
my_file = open("C:\\Users\\Lenovo\\Desktop\\Magisterka\\players_columns.txt", "r")
  
# reading the file
data = my_file.read()
  
# replacing end splitting the text 
# when newline ('\n') is seen.
players_columns = data.split("\n")
my_file.close()
players_columns = [name + '_player' for name in players_columns]
#----------------------------------------------

#function to read and merge seasons of a club

def read_merge_club(club_path,new_colnames):
    os.chdir(club_path)
    seasons = os.listdir()
    df = pd.DataFrame()
    for season in seasons:
        season_path = os.path.join(club_path,season)
        new_season = pd.read_excel(season_path,skiprows=[1,2],header=0)
        try:
            df = pd.concat([df, new_season], axis=0)
        except Exception as e:
            print(season_path)
    df.columns = new_colnames
    return df

#function to format seasons of a club

def format_club(club_df):
    df = club_df.copy()
    df["Date_match"] = df["Date_match"].apply(lambda x:datetime.strptime(x,"%Y-%m-%d"))
    team_df = df.iloc[::2]
    opponent_df = df.iloc[1::2]
    colnames = opponent_df.columns
    opponent_colnames = [name for name in colnames[0:4]]
    for i in range(4,len(colnames)):
        opponent_colnames.append(("Opponent "+colnames[i]))
    opponent_df.columns = opponent_colnames
    df = pd.merge(left = team_df,right = opponent_df,how="inner",on=[name for name in colnames[0:4]],sort=False)
    df = df.sort_values(by="Date_match",ascending = False,ignore_index=True)
    df["Home_match"] = df.loc[:,["Match_match","Team_match"]].apply(lambda x: 1 if x[0].split(" - ")[0]==x[1] else 0,axis=1)
    df["Result_match"] = df.loc[:,["Goals_match","Opponent Goals_match"]].apply(lambda x: np.sign(x[0]-x[1]),axis=1)
    df['Team_match'] = df['Team_match'].apply(lambda x:unidecode(x))
    return df

#function to read and format a player

def read_format_player(player_path,players_columns,player_data,player_name):
    df = pd.read_excel(player_path,header=0)
    df.columns = players_columns
    df["Date_player"] = df["Date_player"].apply(lambda x: datetime.strptime(x,"%Y-%m-%d"),1)
    return df

# wczytywanie i łączenie tabel

# dictionary to change clubs


# wczytywanie klubów

main_path = "C:\\Users\\Lenovo\\Desktop\\Magisterka\\Wyscout\\"
os.chdir(main_path)
os.getcwd()
leagues = os.listdir()
clubs_list = []
club_dfs = []
for league in leagues:
    league_path = os.path.join(main_path,league)
    os.chdir(league_path)
    clubs = os.listdir()
    for club in clubs:
        club_path = os.path.join(league_path,club,"Seasons")
        club_df = read_merge_club(club_path,season_columns)
        df = format_club(club_df)
        clubs_list.append(club)
        club_dfs.append(df)

club_tables = dict(zip(clubs_list, club_dfs))
club_tables['Atletico Madrid']['Team_match']
# wczytywanie zawodników

def is_same_player(players_column,player_name):
    player_name = player_name.lower()
    result1 = (players_column == player_name)
    result2 = players_column.apply(lambda x: (x[0]==player_name[0]) and 
                                    (x.split(" ")[len(x.split(" "))-1]==player_name.split(" ")[len(player_name.split(" "))-1]) and
                                   len(player_name.split(" ")[0])==2)
    result = result1 | result2
    return(result)

i = 1
players_column = player_data["player"].apply(lambda x:x.lower())
player_data_tmp = player_data.copy()
player_data_tmp['tmp'] = 1
main_path = "C:\\Users\\Lenovo\\Desktop\\Magisterka\\Wyscout\\Primera Division\\"
os.chdir(main_path)
os.getcwd()
clubs = os.listdir()
player_stats_list = []
players_wyscout_list = []
for club in clubs:
    club_path = os.path.join(main_path,club,"Players")
    os.chdir(club_path)
    players = os.listdir()
    for player in players:
        player_path = os.path.join(club_path,player)
        player_name = unidecode(player.split("stats ")[1].split(".xlsx")[0])
        df = read_format_player(player_path,players_columns,player_data,player_name)
        df['tmp'] = 1
        try:
            if not player_data_tmp[is_same_player(players_column,player_name)].empty:
                merged = pd.merge(df, player_data_tmp[is_same_player(players_column,player_name)], on=['tmp'])
                merged = merged.drop('tmp', axis=1)
                player_index = merged["Id"][0]
                test_transfer = player_transfers.loc[player_transfers["Id"] == player_index,["Id","next_club","transfer_date"]]
                test_right = test_transfer.sort_values(by="transfer_date")
                test_left = merged.sort_values(by="Date_player") 
                merged = pd.merge_asof(test_left,test_right, left_on = "Date_player", right_on = "transfer_date", direction = "backward")
                injuries = player_injuries.loc[player_injuries["Id"] == player_index,["Id","injury_type","injury_from","injury_to","absence_days"]].sort_values(by="injury_from")
                merged = pd.merge_asof(merged,injuries.drop("injury_to",axis=1), left_on = "Date_player", right_on = "injury_from", direction = "forward")
                is_injured = merged.loc[:,["injury_from"]].groupby(by = "injury_from",sort=False, dropna=False).apply(lambda x:  pd.Series(np.repeat([0], [len(x)], axis=0)).reset_index(drop = True) if all(np.isnat(x)["injury_from"]) else
                                                                                                                                pd.Series(np.repeat([0,1], [len(x)-1,1], axis=0)) ).reset_index(drop = True)
                if is_injured.shape[0]==1:
                    merged["is_injured"] = is_injured.iloc[0,:]
                else:
                    merged["is_injured"] = is_injured
                injury_to = injuries.injury_to.dropna().sort_values() 
                merged = pd.merge_asof(merged,injury_to, left_on = "Date_player", right_on = "injury_to", direction = "backward")
                merged["Age_player"] = (merged["Date_player"] - merged["birth"]).dt.days/365
                merged["Days since last match"] = (merged.Date_player.diff(periods=1).dt.days.fillna(7)).map(int)
                merged["Fit for days"] = (merged.Date_player-merged.injury_to).dt.days.fillna(365).map(int)
                merged["previous_injuries"] = merged.is_injured.cumsum().shift(fill_value=0)
                player_clubs = merged["next_club"].unique().tolist()
                clubs_intersection = [value for value in player_clubs if value in clubs_list]
                if clubs_intersection:
                    conc = pd.concat([club_tables[c] for c in clubs_intersection])
                    merged = pd.merge(merged,conc,how="inner",left_on=["next_club","Date_player"],right_on = ["Team_match","Date_match"],suffixes=('_player', '_match'))
                player_stats_list.append(merged)
                players_wyscout_list.append(player_name)
        except Exception as e:
            print(e)
            print("Error on ",player_name," ",i)
            i = i + 1


#----------------------------

merged.previous_injuries



# łączenie zawodników 



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df_wyscout_players = pd.concat(player_stats_list,ignore_index = True)

df_wyscout_players.shape
df_wyscout_players.columns
print(list(df_wyscout_players.columns))
df_wyscout_players = df_wyscout_players.drop(['Match_player', 'Competition_player', 'Date_player', 'Position_player',  'Id_x'
                                              ,'birth','nationality','position','foot' #,  'player'
                                              ,   'Id_y', 'next_club', 'transfer_date', 'Id','Date_match'
                                              , 'injury_type', 'injury_from', 'absence_days', 'Match_match', 'Competition_match'
                                              , 'Team_match', 'Scheme_match', 'Opponent Team_match', 'Opponent Scheme_match'], axis=1)
df_wyscout_players = df_wyscout_players.dropna(subset=["Goals_match"])
df_wyscout_players = df_wyscout_players.dropna(axis=1).reset_index(drop=True)
df_wyscout_players.iloc[:,74:282] = df_wyscout_players.iloc[:,74:282].applymap(float)
df_wyscout_players.info(verbose=True,show_counts=True)
names = df_wyscout_players['player']
df_wyscout_players.loc[df_wyscout_players['player']=='Antoine Griezmann',:]['Date_player']
df_wyscout_players.loc[df_wyscout_players['player']=='Antoine Griezmann',:]['Match_player']
df_wyscout_players.loc[df_wyscout_players['player']=='Antoine Griezmann',:]['Match_match']
df_wyscout_players.loc[df_wyscout_players['player']=='Antoine Griezmann',:]['next_club']

#normalise every statistic per minute

df_wp = df_wyscout_players.copy()
df_wp.iloc[:,1:7] = df_wp.iloc[:,1:7].div(df_wp.iloc[:,0], axis=0)
df_wp.iloc[:,8:46] = df_wp.iloc[:,8:46].div(df_wp.iloc[:,0], axis=0)
df_wp.iloc[:,47:58] = df_wp.iloc[:,47:58].div(df_wp.iloc[:,0], axis=0)

df_field = df_wp.iloc[:,0:57].copy()

#clustering based on field statistics

scaler = StandardScaler()
df_field_scaled = scaler.fit_transform(df_field)
clustering = AgglomerativeClustering(4).fit(df_field_scaled)
names.value_counts()
clustering_12 =AgglomerativeClustering(12).fit(df_field_scaled)
clustering_8 =AgglomerativeClustering(8).fit(df_field_scaled)

unique, counts = np.unique(clustering.labels_, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(clustering_8.labels_, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(clustering_12.labels_, return_counts=True)
print(np.asarray((unique, counts)).T)

AG_index = np.where(names == 'Antoine Griezmann')
unique, counts = np.unique(clustering.labels_[AG_index], return_counts=True)
print(np.asarray((unique, counts)).T)

SB_index = np.where(names == 'Sergio Busquets')
unique, counts = np.unique(clustering.labels_[SB_index], return_counts=True)
print(np.asarray((unique, counts)).T)

IR_index = np.where(names == 'Ivan Rakitic')
unique, counts = np.unique(clustering.labels_[IR_index], return_counts=True)
print(np.asarray((unique, counts)).T)

JO_index = np.where(names == 'Jan Oblak')
unique, counts = np.unique(clustering.labels_[JO_index], return_counts=True)
print(np.asarray((unique, counts)).T)

LS_index = np.where(names == 'Luis Suarez')
unique, counts = np.unique(clustering.labels_[LS_index], return_counts=True)
print(np.asarray((unique, counts)).T)

KB_index = np.where(names == 'Karim Benzema')
unique, counts = np.unique(clustering.labels_[KB_index], return_counts=True)
print(np.asarray((unique, counts)).T)

JA_index = np.where(names == 'Jordi Alba')
unique, counts = np.unique(clustering.labels_[JA_index], return_counts=True)
print(np.asarray((unique, counts)).T)

DW_index = np.where(names == 'Daniel Wass')
unique, counts = np.unique(clustering.labels_[DW_index], return_counts=True)
print(np.asarray((unique, counts)).T)

GP_index = np.where(names == 'Gerard Pique')
unique, counts = np.unique(clustering.labels_[GP_index], return_counts=True)
print(np.asarray((unique, counts)).T)

K_index = np.where(names == 'Koke')
unique, counts = np.unique(clustering.labels_[K_index], return_counts=True)
print(np.asarray((unique, counts)).T)

US_index = np.where(names == 'Unai Simon')
unique, counts = np.unique(clustering.labels_[US_index], return_counts=True)
print(np.asarray((unique, counts)).T)

def return_unique_counts(player_name,clustering=clustering):
    unique, counts = np.unique(clustering.labels_[names==player_name], return_counts=True)
    arr = np.zeros(shape=(clustering.n_clusters_))
    for i in range(len(unique)):
        arr[unique[i]]=counts[i]/np.sum(counts)
    return(pd.Series(arr))

cluster_counts = pd.DataFrame(names.unique())
cluster_counts.columns = ["name"]
cluster_counts[["group_0","group_1","group_2","group_3"]] = pd.Series(names.unique()).apply(return_unique_counts)

kmeans = KMeans(n_clusters=4, random_state=0).fit(cluster_counts[["group_0","group_1","group_2","group_3"]])
cluster_counts["labels"]=kmeans.labels_
cluster_counts[["name","labels"]]
print(list(cluster_counts["name"][cluster_counts["labels"]==3]))

cluster_8_counts = pd.Series(names.unique()).apply(return_unique_counts,clustering=clustering_8)

kmeans = KMeans(n_clusters=8, random_state=0).fit(cluster_8_counts)
cluster_8_counts["labels"]=kmeans.labels_
unique, counts = np.unique(cluster_8_counts["labels"], return_counts=True)
print(np.asarray((unique, counts)).T)
for i in range(8):
    print(i," ",list(cluster_counts["name"][cluster_8_counts["labels"]==i]))

cluster_12_counts = pd.Series(names.unique()).apply(return_unique_counts,clustering=clustering_12)

kmeans = KMeans(n_clusters=12, random_state=0).fit(cluster_12_counts)
cluster_12_counts["labels"]=kmeans.labels_
unique, counts = np.unique(cluster_12_counts["labels"], return_counts=True)
print(np.asarray((unique, counts)).T)
for i in range(12):
    print(i," ",list(cluster_counts["name"][cluster_12_counts["labels"]==i]))
#injury prediction

X = df_wyscout_players.drop('is_injured', axis=1)
y = df_wyscout_players['is_injured']

y.value_counts()

clf = LogisticRegression(random_state=0).fit(X, y)
predicty = clf.predict(X)
pred_prob = clf.predict_proba(X)
plt.hist(pred_prob,bins=100)
plt.hist(predicty,bins=100)
plt.show()
pd.crosstab(y, predicty)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
clf = LogisticRegression(random_state=0).fit(scaled_X, y)
predicty = clf.predict(scaled_X)
pred_prob = clf.predict_proba(scaled_X)
plt.hist(pred_prob,bins=100)
plt.hist(predicty,bins=100)
plt.show()
pd.crosstab(y, predicty)

clf = KNeighborsClassifier(n_neighbors=15,weights='distance')
clf.fit(scaled_X,y)
predicty = clf.predict(scaled_X)
pred_prob = clf.predict_proba(scaled_X)
plt.hist(pred_prob,bins=100)
plt.hist(predicty,bins=100)
plt.show()
pd.crosstab(y, predicty)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
clf = KNeighborsClassifier(n_neighbors=15,weights='distance')
clf.fit(scaled_X_train,y_train)
predicty = clf.predict(scaled_X_test)
pred_prob = clf.predict_proba(scaled_X_test)
pd.crosstab(y_test, predicty)


clf = DecisionTreeClassifier(random_state=0,max_depth=10,min_samples_split=5,criterion='entropy')
clf.fit(X_train,y_train)
predicty = clf.predict(X_test)
pd.crosstab(y_test, predicty)

fig = plt.figure(figsize=(25,20))
_ = plot_tree(clf, 
                   feature_names=df_wyscout_players.columns,  
                   class_names=["fit","injured"],
                   filled=True)
plt.show()

import graphviz
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph




# clubs comparison

Barca = club_tables["Barcelona"].copy()

print(Barca.columns.tolist()[111:214])
# 0-3 general info
# 4-5 useless info
# 6-108 team stats
# 109-110 useless info
# 111-213 opponent info
# 214-215 home and result

len(Barca.columns.tolist())
len(['Goals_match', 'xG_match', 'Shots_match', 'Shots on target_match', 'Percentage of Shots on target_match', 'Passes_match', 'Passes accurate_match', 'Percentage of Passes accurate_match', 'Possession, %_match', 'Losses_match', 'Losses Low_match', 'Losses Medium_match', 'Losses High_match', 'Recoveries_match', 'Recoveries Low_match', 'Recoveries Medium_match', 'Recoveries High_match', 'Duels_match', 'Duels won_match', 'Percentage of Duels won_match', 'Shots from outside penalty area_match', 'Shots from outside penalty area on target_match', 'Percentage of Shots from outside penalty area on target_match', 'Positional attacks_match', 'Positional attacks with shots_match', 'Percentage of Positional attacks with shots_match', 'Counterattacks_match', 'Counterattacks with shots_match', 'Percentage of Counterattacks with shots_match', 'Set pieces_match', 'Set pieces with shots_match', 'Percentage of Set pieces with shots_match', 'Corners_match', 'Corners with shots_match', 'Percentage of Corners with shots_match', 'Free kicks_match', 'Free kicks with shots_match', 'Percentage of Free kicks with shots_match', 'Penalties_match', 'Penalties converted_match', 'Percentage of Penalties converted_match', 'Crosses_match', 'Crosses accurate_match', 'Percentage of Crosses accurate_match', 'Deep completed crosses_match', 'Deep completed passes_match', 'Penalty area entries _match', 'Penalty area entries (runs)_match', 'Penalty area entries (crosses)_match', 'Touches in penalty area_match', 'Offensive duels_match', 'Offensive duels won_match', 'Percentage of Offensive duels won_match', 'Offsides_match', 'Conceded goals_match', 'Shots against_match', 'Shots against on target_match', 'Percentage of Shots against on target_match', 'Defensive duels_match', 'Defensive duels won_match', 'Percentage of Defensive duels won_match', 'Aerial duels_match', 'Aerial duels won_match', 'Percentage of Aerial duels won_match', 'Sliding tackles_match', 'Sliding tackles successful_match', 'Percentage of Sliding tackles successful_match', 'Interceptions_match', 'Clearances_match', 'Fouls_match', 'Yellow cards_match', 'Red cards_match', 'Forward passes_match', 'Forward passes accurate_match', 'Percentage of Forward passes accurate_match', 'Back passes_match', 'Back passes accurate_match', 'Percentage of Back passes accurate_match', 'Lateral passes_match', 'Lateral passes accurate_match', 'Percentage of Lateral passes accurate_match', 'Long passes_match', 'Long passes accurate_match', 'Percentage of Long passes accurate_match', 'Passes to final third_match', 'Passes to final third accurate_match', 'Percentage of Passes to final third accurate_match', 'Progressive passes_match', 'Progressive passes accurate_match', 'Percentage of Progressive passes accurate_match', 'Smart passes_match', 'Smart passes accurate_match', 'Percentage of Smart passes accurate_match', 'Throw ins_match', 'Throw ins accurate_match', 'Percentage of Throw ins accurate_match', 'Goal kicks_match', 'Match tempo_match', 'Average passes per possession_match', 'Long pass %_match', 'Average shot distance_match', 'Average pass length_match', 'PPDA_match'])


# HSIC implementation

def HSIC(X,Y,kernel_X,kernel_Y):
    n = X.shape[0]
    m_X = X.shape[1]
    m_Y = Y.shape[1]
    H = np.identity(n)-np.ones((n,n))/n
    K_x = np.empty((n,n))
    K_y = np.empty((n,n))
    dist_X = pairwise_distances(X)[np.triu_indices(n, k = 1)]
    dist_Y = pairwise_distances(Y)[np.triu_indices(n, k = 1)]
    med_X = np.median(dist_X)
    med_Y = np.median(dist_Y)
    k_X = kernel_X(med_X)
    k_Y = kernel_Y(med_Y)
    for i in range(n):
        for j in range(n):
            K_x[i,j] = k_X.__call__(np.reshape(X[i,:],(1,m_X)),np.reshape(X[j,:],(1,m_X)))
            K_y[i,j] = k_Y.__call__(np.reshape(Y[i,:],(1,m_Y)),np.reshape(Y[j,:],(1,m_Y)))
    hsic = np.trace(H@K_x@H@K_y)/(n-1)**2
    return(hsic)

X = np.array(Barca.iloc[:,6:109])
Y = np.array(Barca.iloc[:,111:214])

HSIC(X,Y,RBF,RBF)
HSIC(X,X,RBF,RBF)
HSIC(Y,Y,RBF,RBF)

HSIC(X,Y,RBF,RBF)/np.sqrt(HSIC(X,X,RBF,RBF))/np.sqrt(HSIC(Y,Y,RBF,RBF))

def compute_K(X,kernel_X):
    n = X.shape[0]
    m_X = X.shape[1]
    H = np.identity(n)-np.ones((n,n))/n
    K_x = np.empty((n,n))
    dist_X = pairwise_distances(X)[np.triu_indices(n, k = 1)]
    med_X = np.median(dist_X)
    k_X = kernel_X(med_X)
    for i in range(n):
        for j in range(n):
            K_x[i,j] = k_X.__call__(np.reshape(X[i,:],(1,m_X)),np.reshape(X[j,:],(1,m_X)))
    K_x = H@K_x@H
    return(K_x)

def HSCONIC(X,Y,Z,kernel,eps=10e-8):
    n = X.shape[0]
    m_X = X.shape[1]
    m_Y = Y.shape[1]
    m_Z = Z.shape[1]
    H = np.identity(n)-np.ones((n,n))/n
    K_x = np.empty((n,n))
    K_y = np.empty((n,n))
    K_z = np.empty((n,n))
    dist_X = pairwise_distances(X)[np.triu_indices(n, k = 1)]
    dist_Y = pairwise_distances(Y)[np.triu_indices(n, k = 1)]
    dist_Z = pairwise_distances(Z)[np.triu_indices(n, k = 1)]
    med_X = np.maximum(np.median(dist_X),eps)
    med_Y = np.maximum(np.median(dist_Y),eps)
    med_Z = np.maximum(np.median(dist_Z),eps)
    k_X = kernel(med_X)
    k_Y = kernel(med_Y)
    k_Z = kernel(med_Z)
    for i in range(n):
        for j in range(n):
            K_x[i,j] = k_X.__call__(np.reshape(X[i,:],(1,m_X)),np.reshape(X[j,:],(1,m_X)))
            K_y[i,j] = k_Y.__call__(np.reshape(Y[i,:],(1,m_Y)),np.reshape(Y[j,:],(1,m_Y)))
            K_z[i,j] = k_Z.__call__(np.reshape(Z[i,:],(1,m_Z)),np.reshape(Z[j,:],(1,m_Z)))
    K_x = H@K_x@H
    K_y = H@K_y@H
    K_z = H@K_z@H
    M = K_z@np.linalg.matrix_power(K_z+eps*np.identity(n), -2)@K_z
    hsconic = np.trace(K_x@K_y - 2*K_x@M@K_y + K_x@M@K_y@M)/(n-1)**2
    return(hsconic)

def median_of_dist(X):
    n = X.shape[0]
    dist_X = pairwise_distances(X)[np.triu_indices(n, k = 1)]
    med_X = np.median(dist_X)
    return(med_X)

def KCC_simple(X,Z,k,kernel):
    n = X.shape[0]
    U_start = np.zeros((n,k))
    U_start = np.random.rand(n,k)
    for i in range(n):
        U_start[i,i]=1
    def cons_f(U):
        u = U.reshape((n,k))
        return (np.transpose(u)@u).reshape((k*k))
    flat_I = np.identity(k).reshape((k*k))
    nonlinear_constraint = NonlinearConstraint(cons_f,flat_I,flat_I)
    def fun_to_minimize(U):
        u = U.reshape((n,k))
        return(-1*HSCONIC(u,X,Z,kernel=kernel))
    res = minimize(fun_to_minimize, U_start.reshape((n*k)), method='trust-constr',
               constraints=[nonlinear_constraint])
    return res.x.reshape((n,k)) 

X = df_field_scaled[:,:20]
Z = df_field_scaled[:,20:]
k = 8

X = np.random.rand(20,15)
Y = np.random.rand(20,10)
Z = np.random.rand(20,12)
k = 2
kernel=RBF
HSCONIC(X,Y,Z,kernel,eps=10e-8)


KCC_simple(X,Z,k,RBF)