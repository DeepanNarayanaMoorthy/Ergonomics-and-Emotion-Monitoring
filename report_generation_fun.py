import matplotlib.pylab as plt
import json
import datetime

def generatereport(data):
  def infer(s_date,e_date):
      t={'angry':[],'happy':[],'surprised':[],'neutral':[],'sad':[],'scared':[]}
      for i in get_ind:
          t['angry'].append(angry[i])
          t['happy'].append(happy[i])
          t['surprised'].append(surprised[i])
          t['neutral'].append(neutral[i])
          t['sad'].append(sad[i])
          t['scared'].append(scared[i])
      n=len(get_ind)
      l=['angry','happy','surprised','neutral','sad','scared']
      for key,val in t.items():
          d=0
          for i in val:
             if(i>0.4):
                 d+=1
          dataGet[key]=d/n
      names=list(dataGet.keys())
      values=list(dataGet.values())
      plt.bar(range(len(dataGet)),values,tick_label=names)
      plt.savefig('barchart.png',dpi=400)
      plt.show()

  def convertTime():
      for i in data['timedataa']:
          epoch_time = i
          date_time = datetime.datetime.fromtimestamp( epoch_time )  
          time.append(date_time)

  def dataCollect():
      for i in data['angry']:
          angry.append(i)
      for i in data['happy']:
          happy.append(i)
      for i in data['surprised']:
          surprised.append(i)
      for i in data['neutral']:
          neutral.append(i)
      for i in data['sad']:
          sad.append(i)
      for i in data['scared']:
          scared.append(i)

  def inference(type):
    count=0
    flag=0
    timeVal=[]
    if type==1:
      for i in get_ind:
        if data['neck_inclination'][i]>28:
          count+=1
        timeVal.append(time[i])
        neck.append(data['neck_inclination'][i])
      if count==len(get_ind)/2:
        flag=1
      plt.plot(timeVal,neck)
      plt.savefig('linechart1.png')
      plt.show()
    elif type==2:
      for i in get_ind:
        if data['torso_inclination'][i]>8:
          count+=1
        timeVal.append(time[i])
        torso.append(data['torso_inclination'][i])
      if count==len(get_ind)/2:
          flag=1
      plt.plot(timeVal,torso)  
      plt.savefig('linechart2.png')
      plt.show() 
    return flag

  def drowsiness():
    count=0
    flag=0
    timeVal=[]
    for i in get_ind:
        if data['drowsiness_score'][i] > 4:
          count+=1
        timeVal.append(time[i])
        drowsy.append(data['drowsiness_score'][i])
    if count==len(get_ind)/2:
      flag=1
    plt.scatter(timeVal,drowsy)
    plt.savefig('scatterplot.png')
    plt.show()
    return flag

  def getIndex():
    for i in range(0,len(time)):
        if time[i]>s_date and time[i]<e_date:
            get_ind.append(i)

  # f=open('jsondata.txt')
  time=[]
  neck=[]
  torso=[]
  drowsy=[]
  get_ind=[]
  angry=[]
  happy=[]
  surprised=[]
  neutral=[]
  sad=[]
  scared=[]
  dataGet={}
  # data={"timedataa": [1655905835.898674, 1655905836.2186744, 1655905836.554753, 1655905836.8737855, 1655905837.2108657], "angry": [0.028052526, 0.05529974, 0.06660403, 0.03480012, 0.042934477], "scared": [0.05360156, 0.11929174, 0.12533213, 0.06200477, 0.06341797], "happy": [0.001959382, 0.0019642594, 0.0019402344, 0.0027830706, 0.0017529961], "sad": [0.17062306, 0.23376366, 0.30682296, 0.27529377, 0.21328765], "neutral": [0.7413039, 0.5839874, 0.49332088, 0.62301064, 0.67564404], "surprised": [0.0044595483, 0.005693131, 0.005979621, 0.0021076729, 0.0029628891], "neck_inclination": [35, 36, 36, 36, 36], "torso_inclination": [11, 10, 10, 10, 10], "drowsiness_score": [0, 0, 0, 0, 0]}
  # data=json.load(f)
  s_time_data = "20/06/22 18:54:02.2616"
  format_data = "%d/%m/%y %H:%M:%S.%f"
  s_date = datetime.datetime.strptime(s_time_data, format_data)
  e_time_data = "22/06/22 18:54:04.2620"
  e_date = datetime.datetime.strptime(e_time_data, format_data)
  convertTime()
  getIndex()
  dataCollect()
  infer(s_date,e_date)
  flag1=inference(1)
  flag2=inference(2)
  flag3=drowsiness()
  # f.close()

#   !pip install fpdf

  from fpdf import FPDF
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font('Arial','B',16)
  pdf.cell(40,10,"Report")
  pdf.ln(20)
  pdf.set_font('Arial','B',12)
  pdf.cell(20,10,"The following report is for the data between "+s_time_data[:17]+" and "+e_time_data[:17])
  pdf.ln(10)
  pdf.set_font('Arial','B',10)
  pdf.cell(20,1,"Emotion")
  pdf.ln(5)
  pdf.image('barchart.png',w=100,h=60)
  pdf.ln(10)
  pdf.cell(20,1,"Neck inclination")
  pdf.ln(5)
  pdf.image('linechart1.png',w=100,h=60)
  pdf.ln(2)
  if flag1==1:
    pdf.set_font('Arial','',8)
    pdf.cell(20,1,"Your neck is not properly inclined")
  else:
    pdf.set_font('Arial','',8)
    pdf.cell(20,1,"Your neck is properly inclined")
  pdf.ln(10)
  pdf.set_font('Arial','B',10)
  pdf.cell(20,1,"Torso inclination")
  pdf.ln(5)
  pdf.image('linechart2.png',w=100,h=60)
  pdf.ln(2)
  if flag2==1:
    pdf.set_font('Arial','',8)
    pdf.cell(20,1,"Your torso is not properly inclined")
  else:
    pdf.set_font('Arial','',8)
    pdf.cell(20,1,"Your torso is properly inclined")
  pdf.ln(15)
  pdf.set_font('Arial','B',10)
  pdf.cell(20,1,"Drowsiness Score")
  pdf.ln(5)
  pdf.image('scatterplot.png',w=100,h=60)
  pdf.ln(2)
  if flag3==1:
    pdf.set_font('Arial','',8)
    pdf.cell(20,1,"You are too drowsy")
  else:
    pdf.set_font('Arial','',8)
    pdf.cell(20,1,"You are active")
  pdf.output('document.pdf','F')

