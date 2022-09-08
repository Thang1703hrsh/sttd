from http import client
import pymongo 
from pymongo import MongoClient
import pandas as pd
import json
import openpyxl

client = pymongo.MongoClient("mongodb+srv://ducthang1703:Thang1703@cluster0.5fp6wg4.mongodb.net/?retryWrites=true&w=majority")
uploaded_file = 'Line Balancing_SE.xlsm'
df = pd.read_excel(uploaded_file, sheet_name= 'hat1' ,skiprows=3,usecols='B:G')

data = df.to_dict(orient= "record")

db = cluster['ttd_lb']
collection = db["ttd_lb"]

collection.insert_many(data)

# db = client['LineBalance']
# print(db)
