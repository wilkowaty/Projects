import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from geopy.geocoders import Nominatim
import reverse_geocode
import math
import folium
from folium.plugins import MarkerCluster
import time
import re
import string
from folium import plugins
from folium.plugins import FloatImage
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pycountry
import branca.colormap as cmp

def GetCoords(place):
    '''
    przyjmuje jako parametr place miejsce na świecie w postaci "[ulica],[miasto],[państwo]", przy czym wystarczy jedno z tych trzech
    zwraca tuple postaci ([szerokość geograficzna],[długość geograficzna])
    gdy nie jest możliwa identyfikacja zwróci (nan,nan)
    '''
    geolocator = Nominatim(user_agent="projekt3")
    if place=="nan":
        return (float('nan'),float('nan'))
    else:
        try:
            location = geolocator.geocode(place)
            return((location.latitude,location.longitude))
        except:
            return (float('nan'),float('nan'))

def GetCountry(place):
    '''
    przyjmuje jako parametr place miejsce na świecie w postaci "[ulica],[miasto],[państwo]", przy czym wystarczy jedno z tych trzech
    zwraca nazwę państwa po angielsku (string) do którego należą współrzędne, jeśli nie ma takiego zwróci 'NoMatchingCountry'
    '''
    coords=GetCoords(place)
    geolocator = Nominatim(user_agent="projekt3")
    if not math.isnan(coords[0]):
        location = geolocator.reverse(coords,language='en')
        return(location.raw['address']['country'])
    else:
        return('NoMatchingCountry')

def xml2csv(fname, delcols=[]):
    tree = ET.parse(fname)
    root = tree.getroot()
    d = pd.DataFrame([e.attrib for e in root])
    for name in delcols: del d[name]
    d.to_csv(fname + ".csv", index=False)

os.chdir("C:/Users/Lenovo/Desktop/Python/Projekt3/")

# Buddhism
xml2csv("Buddhism/Badges.xml")
xml2csv("Buddhism/Comments.xml")
xml2csv("Buddhism/PostHistory.xml")
xml2csv("Buddhism/PostLinks.xml")
xml2csv("Buddhism/Posts.xml")
xml2csv("Buddhism/Tags.xml")
xml2csv("Buddhism/Users.xml")
xml2csv("Buddhism/Votes.xml")

# Christianity
xml2csv("Christianity/Badges.xml")
xml2csv("Christianity/Comments.xml")
xml2csv("Christianity/PostHistory.xml")
xml2csv("Christianity/PostLinks.xml")
xml2csv("Christianity/Posts.xml")
xml2csv("Christianity/Tags.xml")
xml2csv("Christianity/Users.xml")
xml2csv("Christianity/Votes.xml")

# Hinduism
xml2csv("Hinduism/Badges.xml")
xml2csv("Hinduism/Comments.xml")
xml2csv("Hinduism/PostHistory.xml")
xml2csv("Hinduism/PostLinks.xml")
xml2csv("Hinduism/Posts.xml")
xml2csv("Hinduism/Tags.xml")
xml2csv("Hinduism/Users.xml")
xml2csv("Hinduism/Votes.xml")

# Islam
xml2csv("Islam/Badges.xml")
xml2csv("Islam/Comments.xml")
xml2csv("Islam/PostHistory.xml")
xml2csv("Islam/PostLinks.xml")
xml2csv("Islam/Posts.xml")
xml2csv("Islam/Tags.xml")
xml2csv("Islam/Users.xml")
xml2csv("Islam/Votes.xml")

# Judaism
xml2csv("Judaism/Badges.xml")
xml2csv("Judaism/Comments.xml")
xml2csv("Judaism/PostHistory.xml")
xml2csv("Judaism/PostLinks.xml")
xml2csv("Judaism/Posts.xml")
xml2csv("Judaism/Tags.xml")
xml2csv("Judaism/Users.xml")
xml2csv("Judaism/Votes.xml")


#Buddhism
Badges_b = pd.read_csv("Buddhism/Badges.xml.csv")
Comments_b = pd.read_csv("Buddhism/Comments.xml.csv")
PostHistory_b = pd.read_csv("Buddhism/PostHistory.xml.csv")
PostLinks_b = pd.read_csv("Buddhism/PostLinks.xml.csv")
Posts_b = pd.read_csv("Buddhism/Posts.xml.csv")
Tags_b = pd.read_csv("Buddhism/Tags.xml.csv")
Users_b = pd.read_csv("Buddhism/Users.xml.csv")
Votes_b = pd.read_csv("Buddhism/Votes.xml.csv")

#Christianity
Badges_c = pd.read_csv("Christianity/Badges.xml.csv")
Comments_c = pd.read_csv("Christianity/Comments.xml.csv")
PostHistory_c = pd.read_csv("Christianity/PostHistory.xml.csv")
PostLinks_c = pd.read_csv("Christianity/PostLinks.xml.csv")
Posts_c = pd.read_csv("Christianity/Posts.xml.csv")
Tags_c = pd.read_csv("Christianity/Tags.xml.csv")
Users_c = pd.read_csv("Christianity/Users.xml.csv")
Votes_c = pd.read_csv("Christianity/Votes.xml.csv")

#Hinduism
Badges_h = pd.read_csv("Hinduism/Badges.xml.csv")
Comments_h = pd.read_csv("Hinduism/Comments.xml.csv")
PostHistory_h = pd.read_csv("Hinduism/PostHistory.xml.csv")
PostLinks_h = pd.read_csv("Hinduism/PostLinks.xml.csv")
Posts_h = pd.read_csv("Hinduism/Posts.xml.csv")
Tags_h = pd.read_csv("Hinduism/Tags.xml.csv")
Users_h = pd.read_csv("Hinduism/Users.xml.csv")
Votes_h = pd.read_csv("Hinduism/Votes.xml.csv")

#Islam
Badges_i = pd.read_csv("Islam/Badges.xml.csv")
Comments_i = pd.read_csv("Islam/Comments.xml.csv")
PostHistory_i = pd.read_csv("Islam/PostHistory.xml.csv")
PostLinks_i = pd.read_csv("Islam/PostLinks.xml.csv")
Posts_i = pd.read_csv("Islam/Posts.xml.csv")
Tags_i = pd.read_csv("Islam/Tags.xml.csv")
Users_i = pd.read_csv("Islam/Users.xml.csv")
Votes_i = pd.read_csv("Islam/Votes.xml.csv")

#Judaism
Badges_j = pd.read_csv("Judaism/Badges.xml.csv")
Comments_j = pd.read_csv("Judaism/Comments.xml.csv")
PostHistory_j = pd.read_csv("Judaism/PostHistory.xml.csv")
PostLinks_j = pd.read_csv("Judaism/PostLinks.xml.csv")
Posts_j = pd.read_csv("Judaism/Posts.xml.csv")
Tags_j = pd.read_csv("Judaism/Tags.xml.csv")
Users_j = pd.read_csv("Judaism/Users.xml.csv")
Votes_j = pd.read_csv("Judaism/Votes.xml.csv")


###############################################################################################
#INFROMACJE O UZYTKOWNIKACH
Users_b['Religion'] = 'b'
Users_c['Religion'] = 'c'
Users_h['Religion'] = 'h'
Users_i['Religion'] = 'i'
Users_j['Religion'] = 'j'

Users = pd.concat([Users_b, Users_c, Users_h, Users_i, Users_j])
#lokalizacja wedlug religii
Locations = Users.loc[:,['Religion', 'Location']].dropna(subset=['Location'])
Locations = Locations.reset_index(drop=True)
Coordinates = Locations.Location.drop_duplicates()
Coordinates["Coords"] = pd.read_pickle('User_Coords.pkl')

Coordinates["Coords"] = Locations.Location.drop_duplicates().apply(GetCoords)
Coordinates["Coords"].to_pickle('User_Coords.pkl')

Wsp=pd.concat([Locations.Location.drop_duplicates(),Coordinates.Coords],axis=1)
Wsp.columns=["Location","Coords"]
Wsp.iloc[4816,1]=Wsp.iloc[4723,1]
Wsp.to_csv('Wsp.csv')

Locat = Locations.merge(Wsp,left_on=["Location"],right_on=["Location"])

ColoursReligion={"c":"red",
                 "i":"green",
                 "h":"darkblue",
                 "b":"yellow",
                 "j":"lightblue"}


#mapy z użytkownikami z poszczególnych religii

NumbersReligion={"Christian":1,
                 "Muslim":2,
                 "Irreligion":3,
                 "Hindu":4,
                 "Buddhist":5,
                 "Folk religion":6,
                 "Other religion":7,
                 "Jewish":8}

#dodatkowa tabelka z procentami populacji należącymi do danej religii

adres = 'https://en.wikipedia.org/wiki/Religions_by_country'

res = pd.read_html(adres)

order=res[33].columns
AuOc=res[33].append(res[34:37])[order].reset_index(drop=True)
Europe=res[26].append(res[27:32])[order].reset_index(drop=True)
Asia=res[19].append(res[20:23])[order].reset_index(drop=True) 
Americas=res[14].append(res[15:18])[order].reset_index(drop=True)
Africa=res[8].append(res[9:13])[order].reset_index(drop=True)


Europe.iloc[52,0]="Netherlands"


World=AuOc.append([Europe,Asia,Americas,Africa]).reset_index(drop=True)
World=World.loc[World.Country.Country!="Total",:].reset_index(drop=True)
#lista z krajami świata

Countries=pd.read_table("C:/Users/Lenovo/Desktop/Python/Projekt3/countries.txt",header=None)
Countries=Countries.loc[:,0].tolist()
Countries
#czyścimy tabelkę 

World=World.loc[World.Country.Country.isin(Countries),:].reset_index(drop=True)
World=World.iloc[:,list([0])+list(range(len(World.columns))[3::2])]
World.columns=World.columns.get_level_values(0)


World.iloc[:,1:]=World.iloc[:,1:].applymap(str)

for i in range(1,len(World.columns)):
    World.iloc[:,i]=World.iloc[:,i].str.extract(r"(\d+(?:\.\d+)?)")

World.iloc[:,1:]=World.iloc[:,1:].astype('float')

DominantReligion=World.iloc[:,1:].apply(np.argmax,axis=1).apply(lambda x:World.columns[x+1])
DominantReligion=DominantReligion.append(pd.Series(["Irreligion","Irreligion","Irreligion","Irreligion","Buddhist","Folk religion"])).reset_index(drop=True)
					
ColoursReligion={"Christian":"red",
                 "Muslim":"green",
                 "Irreligion":"grey",
                 "Hindu":"darkblue",
                 "Buddhist":"yellow",
                 "Folk religion":"brown",
                 "Other religion":"black",
                 "Jewish":"lightblue"}

Colours=DominantReligion.apply(lambda x:ColoursReligion[x]).reset_index(drop=True)

#Niestety kilka państw trzeba dodać ręcznie

CountryReligion=pd.concat([World.Country.append(pd.Series(["China","Japan","South Korea","North Korea","Mongolia","Taiwan"])).reset_index(drop=True),
                           DominantReligion,Colours],axis=1)

CountryReligion.columns=["Country","Religion","Colour"]



CountryReligion["Coords"]=CountryReligion.Country.apply(GetCoords)


CountryReligion["Numbers"]=DominantReligion.apply(lambda x:NumbersReligion[x])

#słownik przekształcający nazwę państwa na kod 2-literowy ISO

ISO = {'Afghanistan': 'AF',
 'Albania': 'AL',
 'Algeria': 'DZ',
 'American Samoa': 'AS',
 'Andorra': 'AD',
 'Angola': 'AO',
 'Anguilla': 'AI',
 'Antarctica': 'AQ',
 'Antigua and Barbuda': 'AG',
 'Argentina': 'AR',
 'Armenia': 'AM',
 'Aruba': 'AW',
 'Australia': 'AU',
 'Austria': 'AT',
 'Azerbaijan': 'AZ',
 'Bahamas': 'BS',
 'Bahrain': 'BH',
 'Bangladesh': 'BD',
 'Barbados': 'BB',
 'Belarus': 'BY',
 'Belgium': 'BE',
 'Belize': 'BZ',
 'Benin': 'BJ',
 'Bermuda': 'BM',
 'Bhutan': 'BT',
 'Bolivia': 'BO',
 'Bonaire, Sint Eustatius and Saba': 'BQ',
 'Bosnia and Herzegovina': 'BA',
 'Botswana': 'BW',
 'Bouvet Island': 'BV',
 'Brazil': 'BR',
 'British Indian Ocean Territory': 'IO',
 'Brunei': 'BN',
 'Bulgaria': 'BG',
 'Burkina Faso': 'BF',
 'Burundi': 'BI',
 'Cambodia': 'KH',
 'Cameroon': 'CM',
 'Canada': 'CA',
 'Cape Verde': 'CV',
 'Cayman Islands': 'KY',
 'Central African Republic': 'CF',
 'Chad': 'TD',
 'Chile': 'CL',
 'China': 'CN',
 'Christmas Island': 'CX',
 'Cocos (Keeling) Islands': 'CC',
 'Colombia': 'CO',
 'Comoros': 'KM',
 'Congo, Republic of the': 'CG',
 'Congo, Democratic Republic of the': 'CD',
 'Cook Islands': 'CK',
 'Costa Rica': 'CR',
 'Country name': 'Code',
 'Croatia': 'HR',
 'Cuba': 'CU',
 'Curaçao': 'CW',
 'Cyprus': 'CY',
 'Czech Republic': 'CZ',
 'Ivory Coast': 'CI',
 'Denmark': 'DK',
 'Djibouti': 'DJ',
 'Dominica': 'DM',
 'Dominican Republic': 'DO',
 'Ecuador': 'EC',
 'Egypt': 'EG',
 'El Salvador': 'SV',
 'Equatorial Guinea': 'GQ',
 'Eritrea': 'ER',
 'Estonia': 'EE',
 'Ethiopia': 'ET',
 'Falkland Islands (Malvinas)': 'FK',
 'Faroe Islands': 'FO',
 'Fiji': 'FJ',
 'Finland': 'FI',
 'France': 'FR',
 'French Guiana': 'GF',
 'French Polynesia': 'PF',
 'French Southern Territories': 'TF',
 'Gabon': 'GA',
 'Gambia': 'GM',
 'Georgia': 'GE',
 'Germany': 'DE',
 'Ghana': 'GH',
 'Gibraltar': 'GI',
 'Greece': 'GR',
 'Greenland': 'GL',
 'Grenada': 'GD',
 'Guadeloupe': 'GP',
 'Guam': 'GU',
 'Guatemala': 'GT',
 'Guernsey': 'GG',
 'Guinea': 'GN',
 'Guinea-Bissau': 'GW',
 'Guyana': 'GY',
 'Haiti': 'HT',
 'Heard Island and McDonald Islands': 'HM',
 'Honduras': 'HN',
 'Hong Kong': 'HK',
 'Hungary': 'HU',
 'Iceland': 'IS',
 'India': 'IN',
 'Indonesia': 'ID',
 'Iran': 'IR',
 'Iraq': 'IQ',
 'Ireland': 'IE',
 'Isle of Man': 'IM',
 'Israel': 'IL',
 'Italy': 'IT',
 'Jamaica': 'JM',
 'Japan': 'JP',
 'Jersey': 'JE',
 'Jordan': 'JO',
 'Kazakhstan': 'KZ',
 'Kenya': 'KE',
 'Kiribati': 'KI',
 "North Korea": 'KP',
 'South Korea': 'KR',
 'Kosovo':'XK',
 'Kuwait': 'KW',
 'Kyrgyzstan': 'KG',
 "Laos": 'LA',
 'Latvia': 'LV',
 'Lebanon': 'LB',
 'Lesotho': 'LS',
 'Liberia': 'LR',
 'Libya': 'LY',
 'Liechtenstein': 'LI',
 'Lithuania': 'LT',
 'Luxembourg': 'LU',
 'Macao': 'MO',
 'Macedonia': 'MK',
 'Madagascar': 'MG',
 'Malawi': 'MW',
 'Malaysia': 'MY',
 'Maldives': 'MV',
 'Mali': 'ML',
 'Malta': 'MT',
 'Marshall Islands': 'MH',
 'Martinique': 'MQ',
 'Mauritania': 'MR',
 'Mauritius': 'MU',
 'Mayotte': 'YT',
 'Mexico': 'MX',
 'Micronesia': 'FM',
 'Moldova': 'MD',
 'Monaco': 'MC',
 'Mongolia': 'MN',
 'Montenegro': 'ME',
 'Montserrat': 'MS',
 'Morocco': 'MA',
 'Mozambique': 'MZ',
 'Myanmar': 'MM',
 'Namibia': 'NA',
 'Nauru': 'NR',
 'Nepal': 'NP',
 'Netherlands': 'NL',
 'New Caledonia': 'NC',
 'New Zealand': 'NZ',
 'Nicaragua': 'NI',
 'Niger': 'NE',
 'Nigeria': 'NG',
 'Niue': 'NU',
 'Norfolk Island': 'NF',
 'Northern Mariana Islands': 'MP',
 'Norway': 'NO',
 'Oman': 'OM',
 'Pakistan': 'PK',
 'Palau': 'PW',
 'Palestine': 'PS',
 'Panama': 'PA',
 'Papua New Guinea': 'PG',
 'Paraguay': 'PY',
 'Peru': 'PE',
 'Philippines': 'PH',
 'Pitcairn': 'PN',
 'Poland': 'PL',
 'Portugal': 'PT',
 'Puerto Rico': 'PR',
 'Qatar': 'QA',
 'Romania': 'RO',
 'Russia': 'RU',
 'Rwanda': 'RW',
 'Réunion': 'RE',
 'Saint Barthélemy': 'BL',
 'Saint Helena': 'SH',
 'Saint Kitts and Nevis': 'KN',
 'Saint Lucia': 'LC',
 'Saint Martin': 'MF',
 'Saint Pierre and Miquelon': 'PM',
 'Saint Vincent and the Grenadines': 'VC',
 'Samoa': 'WS',
 'San Marino': 'SM',
 'Sao Tome and Principe': 'ST',
 'Saudi Arabia': 'SA',
 'Senegal': 'SN',
 'Serbia': 'RS',
 'Seychelles': 'SC',
 'Sierra Leone': 'SL',
 'Singapore': 'SG',
 'Sint Maarten (Dutch part)': 'SX',
 'Slovakia': 'SK',
 'Slovenia': 'SI',
 'Solomon Islands': 'SB',
 'Somalia': 'SO',
 'South Africa': 'ZA',
 'South Georgia and the South Sandwich Islands': 'GS',
 'South Sudan': 'SS',
 'Spain': 'ES',
 'Sri Lanka': 'LK',
 'Sudan': 'SD',
 'Suriname': 'SR',
 'Svalbard and Jan Mayen': 'SJ',
 'Swaziland': 'SZ',
 'Sweden': 'SE',
 'Switzerland': 'CH',
 'Syria': 'SY',
 'Taiwan': 'TW',
 'Tajikistan': 'TJ',
 'Tanzania': 'TZ',
 'Thailand': 'TH',
 'Timor-Leste': 'TL',
 'Togo': 'TG',
 'Tokelau': 'TK',
 'Tonga': 'TO',
 'Trinidad and Tobago': 'TT',
 'Tunisia': 'TN',
 'Turkey': 'TR',
 'Turkmenistan': 'TM',
 'Turks and Caicos Islands': 'TC',
 'Tuvalu': 'TV',
 'Uganda': 'UG',
 'Ukraine': 'UA',
 'United Arab Emirates': 'AE',
 'United Kingdom': 'GB',
 'United States': 'US',
 'United States Minor Outlying Islands': 'UM',
 'Uruguay': 'UY',
 'Uzbekistan': 'UZ',
 'Vanuatu': 'VU',
 'Vatican City':'VA',
 'Venezuela': 'VE',
 'Vietnam': 'VN',
 'Virgin Islands, British': 'VG',
 'Virgin Islands, U.S.': 'VI',
 'Wallis and Futuna': 'WF',
 'Western Sahara': 'EH',
 'Yemen': 'YE',
 'Zambia': 'ZM',
 'Zimbabwe': 'ZW',
 'Åland Islands': 'AX'}

#słownik przekształcający 3-literowy kod państwa na 2-literowy

convert_ISO_3166_2_to_3 = {
'SS':'SSD',
'XK':'XXK',
'AF':'AFG',
'AX':'ALA',
'AL':'ALB',
'DZ':'DZA',
'AS':'ASM',
'AD':'AND',
'AO':'AGO',
'AI':'AIA',
'AQ':'ATA',
'AG':'ATG',
'AR':'ARG',
'AM':'ARM',
'AW':'ABW',
'AU':'AUS',
'AT':'AUT',
'AZ':'AZE',
'BS':'BHS',
'BH':'BHR',
'BD':'BGD',
'BB':'BRB',
'BY':'BLR',
'BE':'BEL',
'BZ':'BLZ',
'BJ':'BEN',
'BM':'BMU',
'BT':'BTN',
'BO':'BOL',
'BA':'BIH',
'BW':'BWA',
'BV':'BVT',
'BR':'BRA',
'IO':'IOT',
'BN':'BRN',
'BG':'BGR',
'BF':'BFA',
'BI':'BDI',
'KH':'KHM',
'CM':'CMR',
'CA':'CAN',
'CV':'CPV',
'KY':'CYM',
'CF':'CAF',
'TD':'TCD',
'CL':'CHL',
'CN':'CHN',
'CX':'CXR',
'CC':'CCK',
'CO':'COL',
'KM':'COM',
'CG':'COG',
'CD':'COD',
'CK':'COK',
'CR':'CRI',
'CI':'CIV',
'HR':'HRV',
'CU':'CUB',
'CY':'CYP',
'CZ':'CZE',
'DK':'DNK',
'DJ':'DJI',
'DM':'DMA',
'DO':'DOM',
'EC':'ECU',
'EG':'EGY',
'SV':'SLV',
'GQ':'GNQ',
'ER':'ERI',
'EE':'EST',
'ET':'ETH',
'FK':'FLK',
'FO':'FRO',
'FJ':'FJI',
'FI':'FIN',
'FR':'FRA',
'GF':'GUF',
'PF':'PYF',
'TF':'ATF',
'GA':'GAB',
'GM':'GMB',
'GE':'GEO',
'DE':'DEU',
'GH':'GHA',
'GI':'GIB',
'GR':'GRC',
'GL':'GRL',
'GD':'GRD',
'GP':'GLP',
'GU':'GUM',
'GT':'GTM',
'GG':'GGY',
'GN':'GIN',
'GW':'GNB',
'GY':'GUY',
'HT':'HTI',
'HM':'HMD',
'VA':'VAT',
'HN':'HND',
'HK':'HKG',
'HU':'HUN',
'IS':'ISL',
'IN':'IND',
'ID':'IDN',
'IR':'IRN',
'IQ':'IRQ',
'IE':'IRL',
'IM':'IMN',
'IL':'ISR',
'IT':'ITA',
'JM':'JAM',
'JP':'JPN',
'JE':'JEY',
'JO':'JOR',
'KZ':'KAZ',
'KE':'KEN',
'KI':'KIR',
'KP':'PRK',
'KR':'KOR',
'KW':'KWT',
'KG':'KGZ',
'LA':'LAO',
'LV':'LVA',
'LB':'LBN',
'LS':'LSO',
'LR':'LBR',
'LY':'LBY',
'LI':'LIE',
'LT':'LTU',
'LU':'LUX',
'MO':'MAC',
'MK':'MKD',
'MG':'MDG',
'MW':'MWI',
'MY':'MYS',
'MV':'MDV',
'ML':'MLI',
'MT':'MLT',
'MH':'MHL',
'MQ':'MTQ',
'MR':'MRT',
'MU':'MUS',
'YT':'MYT',
'MX':'MEX',
'FM':'FSM',
'MD':'MDA',
'MC':'MCO',
'MN':'MNG',
'ME':'MNE',
'MS':'MSR',
'MA':'MAR',
'MZ':'MOZ',
'MM':'MMR',
'NA':'NAM',
'NR':'NRU',
'NP':'NPL',
'NL':'NLD',
'AN':'ANT',
'NC':'NCL',
'NZ':'NZL',
'NI':'NIC',
'NE':'NER',
'NG':'NGA',
'NU':'NIU',
'NF':'NFK',
'MP':'MNP',
'NO':'NOR',
'OM':'OMN',
'PK':'PAK',
'PW':'PLW',
'PS':'PSE',
'PA':'PAN',
'PG':'PNG',
'PY':'PRY',
'PE':'PER',
'PH':'PHL',
'PN':'PCN',
'PL':'POL',
'PT':'PRT',
'PR':'PRI',
'QA':'QAT',
'RE':'REU',
'RO':'ROU',
'RU':'RUS',
'RW':'RWA',
'BL':'BLM',
'SH':'SHN',
'KN':'KNA',
'LC':'LCA',
'MF':'MAF',
'PM':'SPM',
'VC':'VCT',
'WS':'WSM',
'SM':'SMR',
'ST':'STP',
'SA':'SAU',
'SN':'SEN',
'RS':'SRB',
'SC':'SYC',
'SL':'SLE',
'SG':'SGP',
'SK':'SVK',
'SI':'SVN',
'SB':'SLB',
'SO':'SOM',
'ZA':'ZAF',
'GS':'SGS',
'ES':'ESP',
'LK':'LKA',
'SD':'SDN',
'SR':'SUR',
'SJ':'SJM',
'SZ':'SWZ',
'SE':'SWE',
'CH':'CHE',
'SY':'SYR',
'TW':'TWN',
'TJ':'TJK',
'TZ':'TZA',
'TH':'THA',
'TL':'TLS',
'TG':'TGO',
'TK':'TKL',
'TO':'TON',
'TT':'TTO',
'TN':'TUN',
'TR':'TUR',
'TM':'TKM',
'TC':'TCA',
'TV':'TUV',
'UG':'UGA',
'UA':'UKR',
'AE':'ARE',
'GB':'GBR',
'US':'USA',
'UM':'UMI',
'UY':'URY',
'UZ':'UZB',
'VU':'VUT',
'VE':'VEN',
'VN':'VNM',
'VG':'VGB',
'VI':'VIR',
'WF':'WLF',
'EH':'ESH',
'YE':'YEM',
'ZM':'ZMB',
'ZW':'ZWE'
}

CountryReligion["Codes"]=World.Country.append(pd.Series(["China","Japan","South Korea","North Korea","Mongolia","Taiwan"])).reset_index(drop=True).apply(lambda x:convert_ISO_3166_2_to_3[ISO[x]])
CountryReligion.to_csv('Country_Religion.csv')
#Judaism


Judaism_Users=Users_j.loc[:,["Location","Id"]].dropna(subset=["Location"]).merge(Wsp,on=["Location"])
Judaism_Users["Coords_with_noise"]=Judaism_Users.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Judaism_Users.to_csv('Judaism_Users.csv')
x=Judaism_Users.Coords_with_noise.apply(lambda x:x[0]).dropna()
y=Judaism_Users.Coords_with_noise.apply(lambda y:y[1]).dropna()

Judaism_Users_Map = go.Figure(data=go.Choropleth(
    locations = CountryReligion['Codes'],
    z = CountryReligion['Numbers'],
    text = CountryReligion[['Country','Religion']],
    colorscale = ["rgba(66,255,248,100)",
                 "rgba(250,0,255,100)",
                 "rgba(127,46,0,100)",
                 "rgba(255,242,0,100)",
                 "rgba(0,25,255,100)",
                 "rgba(132,132,132,100)",
                 "rgba(0,211,52,100)",
                 "rgba(255,0,12,100)"],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='black',
    marker_line_width=0.5,
    showscale=False,
    colorbar_tickprefix = '',
    colorbar_title = '',
    
))


Judaism_Users_Map.add_trace(
    go.Scattergeo(
        lat=x,
        lon=y,
        showlegend=False,
        mode="markers",
        marker=dict(size=5, color="black"),
        hoverinfo="skip")
        )


Judaism_Users_Map.update_layout(
    title_text="Użytkownicy judaistyczni na świecie",
    height=750,
    width=1500,
    template="plotly_white"
)

Judaism_Users_Map.write_html('Judaism_Users_Map.html')

Judaism_Users_Map.show()

#Buddhism


Buddhism_Users=Users_b.loc[:,["Location","Id"]].dropna(subset=["Location"]).merge(Wsp,on=["Location"])
Buddhism_Users["Coords_with_noise"]=Buddhism_Users.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Buddhism_Users.to_csv('Buddhism_Users.csv')
x=Buddhism_Users.Coords_with_noise.apply(lambda x:x[0]).dropna()
y=Buddhism_Users.Coords_with_noise.apply(lambda y:y[1]).dropna()

Buddhism_Users_Map = go.Figure(data=go.Choropleth(
    locations = CountryReligion['Codes'],
    z = CountryReligion['Numbers'],
    text = CountryReligion[['Country','Religion']],
    colorscale = ["rgba(66,255,248,100)",
                 "rgba(250,0,255,100)",
                 "rgba(127,46,0,100)",
                 "rgba(255,242,0,100)",
                 "rgba(0,25,255,100)",
                 "rgba(132,132,132,100)",
                 "rgba(0,211,52,100)",
                 "rgba(255,0,12,100)"],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='black',
    marker_line_width=0.5,
    showscale=False,
    colorbar_tickprefix = '',
    colorbar_title = '',
    
))


Buddhism_Users_Map.add_trace(
    go.Scattergeo(
        lat=x,
        lon=y,
        showlegend=False,
        mode="markers",
        marker=dict(size=5, color="black"),
        hoverinfo="skip")
        )


Buddhism_Users_Map.update_layout(
    title_text="Użytkownicy buddyjscy na świecie",
    height=750,
    width=1500,
    template="plotly_white"
)

Buddhism_Users_Map.write_html('Buddhism_Users_Map.html')

Buddhism_Users_Map.show()

#Hinduism


Hinduism_Users=Users_h.loc[:,["Location","Id"]].dropna(subset=["Location"]).merge(Wsp,on=["Location"])
Hinduism_Users["Coords_with_noise"]=Hinduism_Users.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Hinduism_Users.to_csv('Hinduism_Users.csv')
x=Hinduism_Users.Coords_with_noise.apply(lambda x:x[0]).dropna()
y=Hinduism_Users.Coords_with_noise.apply(lambda y:y[1]).dropna()

Hinduism_Users_Map = go.Figure(data=go.Choropleth(
    locations = CountryReligion['Codes'],
    z = CountryReligion['Numbers'],
    text = CountryReligion[['Country','Religion']],
    colorscale = ["rgba(66,255,248,100)",
                 "rgba(250,0,255,100)",
                 "rgba(127,46,0,100)",
                 "rgba(255,242,0,100)",
                 "rgba(0,25,255,100)",
                 "rgba(132,132,132,100)",
                 "rgba(0,211,52,100)",
                 "rgba(255,0,12,100)"],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='black',
    marker_line_width=0.5,
    showscale=False,
    colorbar_tickprefix = '',
    colorbar_title = '',
    
))


Hinduism_Users_Map.add_trace(
    go.Scattergeo(
        lat=x,
        lon=y,
        showlegend=False,
        mode="markers",
        marker=dict(size=5, color="black"),
        hoverinfo="skip")
        )


Hinduism_Users_Map.update_layout(
    title_text="Użytkownicy hinduistyczni na świecie",
    height=750,
    width=1500,
    template="plotly_white"
)

Hinduism_Users_Map.write_html('Hinduism_Users_Map.html')

Hinduism_Users_Map.show()

#Christianity


Christianity_Users=Users_c.loc[:,["Location","Id"]].dropna(subset=["Location"]).merge(Wsp,on=["Location"])
Christianity_Users["Coords_with_noise"]=Christianity_Users.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Christianity_Users.to_csv('Christianity_Users.csv')
x=Christianity_Users.Coords_with_noise.apply(lambda x:x[0]).dropna()
y=Christianity_Users.Coords_with_noise.apply(lambda y:y[1]).dropna()

Christianity_Users_Map = go.Figure(data=go.Choropleth(
    locations = CountryReligion['Codes'],
    z = CountryReligion['Numbers'],
    text = CountryReligion[['Country','Religion']],
    colorscale = ["rgba(66,255,248,100)",
                 "rgba(250,0,255,100)",
                 "rgba(127,46,0,100)",
                 "rgba(255,242,0,100)",
                 "rgba(0,25,255,100)",
                 "rgba(132,132,132,100)",
                 "rgba(0,211,52,100)",
                 "rgba(255,0,12,100)"],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='black',
    marker_line_width=0.5,
    showscale=False,
    colorbar_tickprefix = '',
    colorbar_title = '',
    
))


Christianity_Users_Map.add_trace(
    go.Scattergeo(
        lat=x,
        lon=y,
        showlegend=False,
        mode="markers",
        marker=dict(size=5, color="black"),
        hoverinfo="skip")
        )


Christianity_Users_Map.update_layout(
    title_text="Użytkownicy chrześcijańscy na świecie",
    height=750,
    width=1500,
    template="plotly_white"
)

Christianity_Users_Map.write_html('Christianity_Users_Map.html')

Christianity_Users_Map.show()

#Islam


Islam_Users=Users_i.loc[:,["Location","Id"]].dropna(subset=["Location"]).merge(Wsp,on=["Location"])
Islam_Users["Coords_with_noise"]=Islam_Users.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Islam_Users.to_csv('Islam_Users.csv')
x=Islam_Users.Coords_with_noise.apply(lambda x:x[0]).dropna()
y=Islam_Users.Coords_with_noise.apply(lambda y:y[1]).dropna()

Islam_Users_Map = go.Figure(data=go.Choropleth(
    locations = CountryReligion['Codes'],
    z = CountryReligion['Numbers'],
    text = CountryReligion[['Country','Religion']],
    colorscale = ["rgba(66,255,248,100)",
                 "rgba(250,0,255,100)",
                 "rgba(127,46,0,100)",
                 "rgba(255,242,0,100)",
                 "rgba(0,25,255,100)",
                 "rgba(132,132,132,100)",
                 "rgba(0,211,52,100)",
                 "rgba(255,0,12,100)"],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='black',
    marker_line_width=0.5,
    showscale=False,
    colorbar_tickprefix = '',
    colorbar_title = '',
    
))


Islam_Users_Map.add_trace(
    go.Scattergeo(
        lat=x,
        lon=y,
        showlegend=False,
        mode="markers",
        marker=dict(size=5, color="black"),
        hoverinfo="skip")
        )


Islam_Users_Map.update_layout(
    title_text="Użytkownicy muzułmańscy na świecie",
    height=750,
    width=1500,
    template="plotly_white"
)

Islam_Users_Map.write_html('Islam_Users_Map.html')

Islam_Users_Map.show()




#################################################################################################################
# info o postach
Posts_b['Religion'] = 'b'
Posts_c['Religion'] = 'c'
Posts_h['Religion'] = 'h'
Posts_i['Religion'] = 'i'
Posts_j['Religion'] = 'j'
Posts = pd.concat([Posts_b, Posts_c, Posts_h, Posts_i, Posts_j])

######################################################################################################################
#info z postow i uzytkownikow

#najwyzej punktowane posty i lokalizacja ich autorow
Posts_Users_b = Posts_b.merge(Users_b, left_on = 'OwnerUserId', right_on = 'Id', suffixes = ('_P', '_U'))
table = Posts_Users_b[Posts_Users_b['PostTypeId'] == 1].dropna(subset = ['Location'])
Buddhism_Posts_TopScore = table[['Location', 'Score', 'Title']].sort_values(by = 'Score', ascending = False)
Buddhism_Posts_TopScore=Buddhism_Posts_TopScore.merge(Wsp,on=["Location"])
Buddhism_Posts_TopScore["Coords_with_noise"]=Buddhism_Posts_TopScore.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Buddhism_Posts_TopScore.to_csv('Buddhism_Posts_TopScore.csv')

frame={'latitude' : Buddhism_Posts_TopScore["Coords_with_noise"].apply(lambda x:x[0]), 'longitude' : Buddhism_Posts_TopScore["Coords_with_noise"].apply(lambda x:x[1])}


world_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)

for i in range(len(Coords)) :
    if i<int(np.floor(len(pd.DataFrame(frame).dropna().to_numpy())/4)): color='green'
    if i>=int(np.floor(len(pd.DataFrame(frame).dropna().to_numpy())/4)) and i<int(np.floor(len(pd.DataFrame(frame).dropna().to_numpy())/2)): color='yellow'
    if i<int(np.floor(3*len(pd.DataFrame(frame).dropna().to_numpy())/4)) and i>=int(np.floor(len(pd.DataFrame(frame).dropna().to_numpy())/2)): color='orange'
    if i>int(np.floor(3*len(pd.DataFrame(frame).dropna().to_numpy())/4)): color='red'
    if not math.isnan(Coords_noised.iloc[i][0]):
        folium.CircleMarker(Coords_noised.iloc[i], radius=4, weight=2, color=color, fill_color=color, fill_opacity=1).add_to(world_map)


image_file = 'Buddhism_Posts_TopScore_legend.PNG'

FloatImage(image_file, bottom=0, left=0).add_to(world_map)

world_map.save('Buddhism_Posts_TopScore.html')


#christianity - najbardziej lubiane posty
Posts_Users_c = Posts_c.merge(Users_c, left_on = 'OwnerUserId', right_on = 'Id', suffixes = ('_P', '_U'))
table = Posts_Users_c[Posts_Users_c['PostTypeId'] == 1].dropna(subset = ['Location'])
Christianity_Posts_TopScore=table[['Location', 'FavoriteCount', 'Title']].sort_values(by = 'FavoriteCount', ascending = False)
Christianity_Posts_TopScore=Christianity_Posts_TopScore.merge(Wsp,on=["Location"])
Christianity_Posts_TopScore["Coords_with_noise"]=Christianity_Posts_TopScore.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Christianity_Posts_TopScore.to_csv('Christianity_Posts_TopScore.csv')

frame_c={'latitude' : Christianity_Posts_TopScore["Coords_with_noise"].apply(lambda x:x[0]), 'longitude' : Christianity_Posts_TopScore["Coords_with_noise"].apply(lambda x:x[1])}



Christianity_Posts_TopScore_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(Christianity_Posts_TopScore_map)

for i in range(len(Christianity_Posts_TopScore["Coords_with_noise"])) :
    if i<int(np.floor(len(pd.DataFrame(frame_c).dropna().to_numpy())/4)): color='green'
    if i>=int(np.floor(len(pd.DataFrame(frame_c).dropna().to_numpy())/4)) and i<int(np.floor(len(pd.DataFrame(frame_c).dropna().to_numpy())/2)): color='yellow'
    if i<int(np.floor(3*len(pd.DataFrame(frame_c).dropna().to_numpy())/4)) and i>=int(np.floor(len(pd.DataFrame(frame_c).dropna().to_numpy())/2)): color='orange'
    if i>int(np.floor(3*len(pd.DataFrame(frame_c).dropna().to_numpy())/4)): color='red'
    if not math.isnan(Christianity_Posts_TopScore["Coords_with_noise"].iloc[i][0]):
        folium.CircleMarker(Christianity_Posts_TopScore["Coords_with_noise"].iloc[i], radius=4, weight=2, color=color, fill_color=color, fill_opacity=1).add_to(Christianity_Posts_TopScore_map)

image_file = 'Christianity_Posts_TopScore_legend.PNG'

FloatImage(image_file, bottom=0, left=0).add_to(Christianity_Posts_TopScore_map)

Christianity_Posts_TopScore_map.save('Christianity_Posts_TopScore.html')

#hinduism - najczęściej komentowane posty
Posts_Users_h = Posts_h.merge(Users_h, left_on = 'OwnerUserId', right_on = 'Id', suffixes = ('_P', '_U'))
table = Posts_Users_h[Posts_Users_h['PostTypeId'] == 1].dropna(subset = ['Location'])
Hinduism_Posts_MostComments=table[['Location', 'CommentCount', 'Title']].sort_values(by = 'CommentCount', ascending = False)
Hinduism_Posts_MostComments=Hinduism_Posts_MostComments.merge(Wsp,on=["Location"])
Hinduism_Posts_MostComments["Coords_with_noise"]=Hinduism_Posts_MostComments.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Hinduism_Posts_MostComments.to_csv('Hinduism_Posts_MostComments.csv')

frame_h={'latitude' : Hinduism_Posts_MostComments["Coords_with_noise"].apply(lambda x:x[0]), 'longitude' : Hinduism_Posts_MostComments["Coords_with_noise"].apply(lambda x:x[1])}

Hinduism_Posts_MostComments_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(Hinduism_Posts_MostComments_map)


for i in range(len(Hinduism_Posts_MostComments["Coords_with_noise"])) :
    if i<int(np.floor(len(pd.DataFrame(frame_h).dropna().to_numpy())/4)): color='green'
    if i>=int(np.floor(len(pd.DataFrame(frame_h).dropna().to_numpy())/4)) and i<int(np.floor(len(pd.DataFrame(frame_h).dropna().to_numpy())/2)): color='yellow'
    if i<int(np.floor(3*len(pd.DataFrame(frame_h).dropna().to_numpy())/4)) and i>=int(np.floor(len(pd.DataFrame(frame_h).dropna().to_numpy())/2)): color='orange'
    if i>int(np.floor(3*len(pd.DataFrame(frame_h).dropna().to_numpy())/4)): color='red'
    if not math.isnan(Hinduism_Posts_MostComments["Coords_with_noise"].iloc[i][0]):
        folium.CircleMarker(Hinduism_Posts_MostComments["Coords_with_noise"].iloc[i], radius=4, weight=2, color=color, fill_color=color, fill_opacity=1).add_to(Hinduism_Posts_MostComments_map)

image_file = 'Hinduism_Posts_MostComments_legend.PNG'

FloatImage(image_file, bottom=0, left=0).add_to(Hinduism_Posts_MostComments_map)

Hinduism_Posts_MostComments_map.save('Hinduism_Posts_MostComments.html')



#islam - data ostatniej edycji
Posts_Users_i = Posts_i.merge(Users_i, left_on = 'OwnerUserId', right_on = 'Id', suffixes = ('_P', '_U'))
table = Posts_Users_i[Posts_Users_i['PostTypeId'] == 1].dropna(subset = ['Location'])
Islam_Posts_LastEditDate = table[['Location', 'LastEditDate', 'Title']].sort_values(by = 'LastEditDate', ascending = False)
Islam_Posts_LastEditDate = Islam_Posts_LastEditDate.dropna(subset=["LastEditDate"]).reset_index(drop=True)
Islam_Posts_LastEditDate.LastEditDate=pd.to_datetime(Islam_Posts_LastEditDate.LastEditDate.astype('datetime64').apply(lambda x:x.date()))
Islam_Posts_LastEditDate=Islam_Posts_LastEditDate.merge(Wsp,on=["Location"])
Islam_Posts_LastEditDate["Coords_with_noise"]=Islam_Posts_LastEditDate.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Islam_Posts_LastEditDate.to_csv('Islam_Posts_LastEditDate.csv')


Time_colormap = cmp.LinearColormap(
    ['red', 'orange', 'yellow','green'],
    vmin=np.min(Islam_Posts_LastEditDate.LastEditDate),
    vmax=np.max(Islam_Posts_LastEditDate.LastEditDate),
    caption='Czas od ostatniej aktywności posta' #Caption for Color scale or Legend
)

Islam_Posts_LastEditDate_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(Islam_Posts_LastEditDate_map)

for i in range(len(Islam_Posts_LastEditDate.Coords_with_noise)) :
    if not math.isnan(Islam_Posts_LastEditDate.Coords_with_noise.iloc[i][0]):
        folium.CircleMarker(Islam_Posts_LastEditDate.Coords_with_noise[i], radius=4, weight=2, 
                            color=Time_colormap(Islam_Posts_LastEditDate.LastEditDate[i]), 
                            fill_color=Time_colormap(Islam_Posts_LastEditDate.LastEditDate[i]), 
                            fill_opacity=1).add_to(Islam_Posts_LastEditDate_map)

image_file = 'Islam_legend.PNG'

FloatImage(image_file, bottom=0, left=0).add_to(Islam_Posts_LastEditDate_map)


Islam_Posts_LastEditDate_map.save('Islam_Posts_LastEditDate.html')








#judaism - liczba odpowiedzi
Posts_Users_j = Posts_j.merge(Users_j, left_on = 'OwnerUserId', right_on = 'Id', suffixes = ('_P', '_U'))
table = Posts_Users_j[Posts_Users_j['PostTypeId'] == 1].dropna(subset = ['Location'])
Judaism_Posts_MostAnswers = table[['Location', 'AnswerCount', 'Title']].sort_values(by = 'AnswerCount', ascending = False)
Judaism_Posts_MostAnswers=Judaism_Posts_MostAnswers.merge(Wsp,on=["Location"])
Judaism_Posts_MostAnswers=Judaism_Posts_MostAnswers.loc[Judaism_Posts_MostAnswers.Coords.apply(lambda x:not math.isnan(x[0])),:].reset_index(drop=True)
Judaism_Posts_MostAnswers["Coords_with_noise"]=Judaism_Posts_MostAnswers.Coords.apply(lambda x: ((x[0]+np.random.normal(0,1/12,1))[0],(x[1]+np.random.normal(0,1/12,1))[0]))
Judaism_Posts_MostAnswers.to_csv('Judaism_Posts_MostAnswers.csv')


frame_j={'latitude' : Judaism_Posts_MostAnswers["Coords_with_noise"].apply(lambda x:x[0]), 'longitude' : Judaism_Posts_MostAnswers["Coords_with_noise"].apply(lambda x:x[1])}



Judaism_Posts_MostAnswers_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(Judaism_Posts_MostAnswers_map)

for i in range(len(Judaism_Posts_MostAnswers["Coords_with_noise"])) :
    if i<int(np.floor(len(pd.DataFrame(frame_j).dropna().to_numpy())/4)): color='green'
    if i>=int(np.floor(len(pd.DataFrame(frame_j).dropna().to_numpy())/4)) and i<int(np.floor(len(pd.DataFrame(frame_j).dropna().to_numpy())/2)): color='yellow'
    if i<int(np.floor(3*len(pd.DataFrame(frame_j).dropna().to_numpy())/4)) and i>=int(np.floor(len(pd.DataFrame(frame_j).dropna().to_numpy())/2)): color='orange'
    if i>int(np.floor(3*len(pd.DataFrame(frame_j).dropna().to_numpy())/4)): color='red'
    if not math.isnan(Judaism_Posts_MostAnswers["Coords_with_noise"].iloc[i][0]):
        folium.CircleMarker(Judaism_Posts_MostAnswers["Coords_with_noise"].iloc[i], radius=4, weight=2, color=color, fill_color=color, fill_opacity=1).add_to(Judaism_Posts_MostAnswers_map)

image_file = 'Judaism_Posts_MostAnswers_legend.PNG'

FloatImage(image_file, bottom=0, left=0).add_to(Judaism_Posts_MostAnswers_map)

Judaism_Posts_MostAnswers_map.save('Judaism_Posts_MostAnswers.html')




