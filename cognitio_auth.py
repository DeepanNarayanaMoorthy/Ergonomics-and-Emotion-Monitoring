import os, sys
import boto3
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
import streamlit as st
from cryptography.fernet import Fernet
import cv2 
# from __future__ import print_function # Python 2/3 compatibility
import json
import decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)


fernatkey=""
trainerfile='./trainer/trainer.yml'

def writeaccesstoken(access_token):
    f = open("accesstoken", "w")
    f.write(access_token)
    f.close()

def writeusername(username):
    f = open("username", "w")
    f.write(username)
    f.close()

def readaccesstoken():
    f = open("accesstoken", "r")
    return(f.read())

def encrypt(filename, key, encfilename):
	f = Fernet(key)
	with open(filename, "rb") as file:
		file_data = file.read()
	encrypted_data = f.encrypt(file_data)
	with open(encfilename, "wb") as file:
		file.write(encrypted_data)
	return

def decrypt(filename, key, decfilename):
	f = Fernet(key)
	with open(filename, "rb") as file:
		encrypted_data = file.read()
	decrypted_data = f.decrypt(encrypted_data)
	with open(decfilename, "wb") as file:
		file.write(decrypted_data)
	return

# st.set_page_config(layout="wide")
def returnclient():
    load_dotenv(find_dotenv())
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    client = boto3.client("cognito-idp", region_name=os.getenv("REGION_NAME"))
    return(client)

def loginpage():
    client=returnclient()
    global  fernatkey, trainerfile
    col1, col2, col3=st.columns(3)

    companyname=col2.text_input("Enter Company Name")
    username=col2.text_input("Enter Username")
    password=col2.text_input("Enter Password", type="password")

    # col21, col22=col2.columns(3)
    loginpressed=col2.button("Login")

    if(loginpressed):
        try:
            response = client.initiate_auth(
            ClientId=os.getenv("COGNITO_USER_CLIENT_ID"),
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": username, "PASSWORD": password},
            )
            access_token = response["AuthenticationResult"]["AccessToken"]
            writeaccesstoken(access_token)
            response = client.get_user(AccessToken=access_token)
            st.info(access_token)
            st.info(response)
            writeusername(username+","+companyname)
            for i in response['UserAttributes']:
                if(i["Name"]=='custom:custom_filehash'):
                    fernatkey=i["Value"]
                    break
            if(os.path.exists(trainerfile)):
                try:
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(trainerfile)
                    st.info("Face Recognition Data Already Decrypted, Proceed to Ergonomic analysis")
                except:
                    try:
                        fernatkey=bytes(fernatkey, 'utf-8')
                        decrypt(trainerfile, fernatkey, trainerfile)
                        recognizer = cv2.face.LBPHFaceRecognizer_create()
                        recognizer.read(trainerfile)
                        st.info("Face Recognition Data Exists, Proceed to Ergonomic analysis")
                    except:
                        st.error("Face Recognition Data corrupted")
                        st.error("Please register your face again before proceeding to Ergonomic analysis")
                        st.error(sys.exc_info())
            else:
                st.error("Face Recognition Data does not exist, please register your face before proceeding to Ergonomic analysis")
        except Exception as e:
            st.error("Please Check your Username or password\n"+str(sys.exc_info()))
            st.error(sys.exc_info())
        loginpressed=False

def forgotpasswordpage():
    client=returnclient()
    col1, col2, col3=st.columns(3)
    username=col2.text_input("Enter Username")
    otpbutton=col2.button("Send OPT to Mail")

    verificationcode=col2.text_input("Verification code")
    newpassword=col2.text_input("New Password")
    resetpasswordbutton=col2.button("Reset Password")

    if(otpbutton):
        try:
            params = {"ClientId": os.getenv("COGNITO_USER_CLIENT_ID"), "Username": username}
            client.forgot_password(**params)
            st.warning("Check your registered email for verification code")
        except:
            st.error("Username does not exist")
            st.error(sys.exc_info())

    if(resetpasswordbutton):
        try:
            params = {
                "ClientId": os.getenv("COGNITO_USER_CLIENT_ID"),
                "Username": username,
                "ConfirmationCode": verificationcode,
                "Password": newpassword,
            }
            response = client.confirm_forgot_password(**params)
            st.info("Password Reset Successful. Please try to sign in again.")
        except:
            st.error("Wrong Verification code. Try Again.")
            st.error(sys.exc_info())
            pass
        resetpasswordbutton=False

def registeruserpage():
    client=returnclient()
    col1, col2, col3=st.columns(3)
    email=col2.text_input("Enter Email address")
    username=col2.text_input("Enter Username")
    password=col2.text_input("Enter Password")
    hashvalue="DUMMY_VAL"
    signupbutton=col2.button("Sign Up")
    verificationcode=col2.text_input("Enter Verification code")
    verifybutton=col2.button("Verify")

    if(signupbutton):
        try:
            response = client.sign_up(
            ClientId=os.getenv("COGNITO_USER_CLIENT_ID"),
            Username=username,
            Password=password,
            UserAttributes=[{"Name": "email", "Value": email}, {"Name": "custom:custom_filehash","Value": hashvalue}],
            )
            st.warning("Please Check your Mail for Verification Code")
        except:
            st.error("Check Username OR Username already exists")
            st.error(sys.exc_info())

    if(verifybutton):
        try:
            params = {
                "ClientId": os.getenv("COGNITO_USER_CLIENT_ID"),
                "Username": username,
                "ConfirmationCode": verificationcode,
            }
            client.confirm_sign_up(**params)
            st.info("Verification Successful")
        except:
            st.warning("Please Enter Correct Verification Code")

def updatedynamo():
    dynamodb = boto3.resource('dynamodb',
        aws_access_key_id=os.getenv("DB_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key= os.getenv("DB_AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION_NAME")
    )

    table = dynamodb.Table("wellnessapptable")

    f = open("analysisdatatemp", "r")
    dictt=json.loads(f.read())
    f.close()
    f=open('username', 'r')

    userdata=f.read()
    f.close()
    userdata=userdata.split(",")
    userid=userdata[0]
    companyname=userdata[1]
    item= {'companyname': companyname, "userid":userid}
    dicttkeys=list(dictt)
    for i in dicttkeys:
        item[i]=dictt[i]
    
    try:
        table.get_item(Key={'companyname': companyname, 'userid': userid})["Item"]
    except:
        st.info("Data does not exist in DynamoDB, New Entry Created")
        emptydictt={'angry': [],
            'companyname': companyname,
            'drowsiness_score': [],
            'happy':[],
            'neck_inclination': [],
            'neutral': [],
            'sad': [],
            'scared': [],
            'surprised': [],
            'timedataa':[],
            'torso_inclination': [],
            'userid': userid}
        item = json.loads(json.dumps(emptydictt), parse_float=decimal.Decimal)
        table.put_item(
            Item=item
        )

    dictt=json.loads(json.dumps(dictt), parse_float=decimal.Decimal)

    print(dictt)
    response = table.update_item(
        Key={
            'companyname': companyname,
            'userid': userid
        },
        UpdateExpression="set happy=list_append(happy, :haplist), \
                            scared=list_append(scared, :scalist), \
                            neutral=list_append(neutral, :neut), \
                            timedataa=list_append(timedataa, :timelist), \
                            torso_inclination=list_append(torso_inclination, :torsoinc), \
                            neck_inclination=list_append(neck_inclination, :necinc), \
                            sad=list_append(sad, :sadlist), \
                            surprised=list_append(surprised, :surp), \
                            angry=list_append(angry, :angrylist), \
                            drowsiness_score=list_append(drowsiness_score, :drowslist)",
        ExpressionAttributeValues={ 
            ':haplist': dictt['happy'],
            ':scalist': dictt['scared'],
            ':neut': dictt['neutral'],
            ':timelist': dictt['timedataa'],
            ':torsoinc': dictt['torso_inclination'],
            ':necinc': dictt['neck_inclination'],
            ':sadlist': dictt['sad'],
            ':surp': dictt['surprised'],
            ':angrylist': dictt['angry'],
            ':drowslist': dictt['drowsiness_score'],
        },
        ReturnValues="UPDATED_NEW"
    )
    st.info(str(response))

def logoutfun():

    client=returnclient()
    global  fernatkey, trainerfile
    col1, col2, col3=st.columns(3)
    logoutbutton=col2.button("Log Out")
    access_token=readaccesstoken()
    if(logoutbutton):
        try:
            updatedynamo()
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(trainerfile)
            fernatkey = Fernet.generate_key()
            fernatkey=fernatkey.decode("utf-8") 
            encrypt(trainerfile, fernatkey, trainerfile)
            client.update_user_attributes(
                UserAttributes=[{'Name': 'custom:custom_filehash', 'Value': fernatkey}], AccessToken=access_token)
            st.info('Face Recognition File Encrypted')
            try:
                os.remove("accesstoken")
            except:
                pass

        except:
            st.error("Face Recognition Data corrupted")
            st.error("Please register your face again before proceeding to Ergonomic analysis")
            st.error(access_token)
            st.error(str(sys.exc_info()))



# page_names_to_funcs = {
#     "Login Page": loginpage,
#     "Register New User": registeruserpage,
#     "Forgot Password": forgotpasswordpage,
#     "Log Out": logoutfun
# }

# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# page_names_to_funcs[selected_page]()