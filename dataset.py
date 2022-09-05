import pandas as pd
import numpy as np 

import streamlit as st 
from streamlit_option_menu import option_menu 

import os 
import streamlit_authenticator as stauth
import datetime 
import database as db 
import openpyxl 

# file_path = r'Line Balancing.xlsm'
# input_data = pd.read_excel(file_path, sheet_name='cap_ultra_test',skiprows=3,usecols='B:H')

path = 'Line Balancing_SE.xlsm'
from openpyxl import load_workbook
wb = load_workbook(filename = path)
g_sheet=wb.sheetnames
print(g_sheet)
# for i in g_sheet:
#     print(i)

# file modification timestamp of a file
m_time = os.path.getmtime(wb)
# convert timestamp into DateTime object
dt_m = datetime.datetime.fromtimestamp(m_time)
print('Chỉnh sửa ngày :', dt_m)

# file creation timestamp in float
c_time = os.path.getctime(wb)
# convert creation timestamp into DateTime object
dt_c = datetime.datetime.fromtimestamp(c_time)
print('Ngày tạo :', dt_c)
