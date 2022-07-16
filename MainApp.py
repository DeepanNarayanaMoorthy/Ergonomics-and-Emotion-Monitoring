import streamlit as st
import base64

# Custom imports 
from multipage import MultiPage
from Combined import mainappfun, facetrainfun
from cognitio_auth import *
# Create an instance of the app 

app = MultiPage()

st.set_page_config(layout="wide")

_, titlebar, _=st.columns(3)
# Title of the main page
titlebar.title("Employee Wellness Application")

app.add_page("Login Page", loginpage)
app.add_page("Register New User", registeruserpage)
app.add_page("Forgot Password", forgotpasswordpage)
app.add_page("Register your Face", facetrainfun)
app.add_page("Start Detection", mainappfun)
app.add_page("Log Out", logoutfun)

# The main app
app.run()