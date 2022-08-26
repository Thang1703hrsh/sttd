from inspect import stack
import pandas as pd
import matplotlib.pyplot as plt 
import plotly.express as px
import xlrd
import openpyxl

book = openpyxl.load_workbook("Line Balancing_SE.xlsm")
sheet = book.sheetnames #Gets the first sheet.

print(sheet)

