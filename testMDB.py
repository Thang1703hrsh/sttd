from http import client
import pymongo 
import pandas as pd
import json
import openpyxl

client = pymongo.MongoClient("mongodb+srv://ducthang1703:Thang1703@cluster0.5fp6wg4.mongodb.net/?retryWrites=true&w=majority")
db = cluster["ttd_lb"]

collection = db["ttd_lb"]

collection.insert_one({})