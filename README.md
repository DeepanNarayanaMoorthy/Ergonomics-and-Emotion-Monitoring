# Ergonomics and Emotion Monitoring

An application developed to improve and notify users about incorrect **posture**. **Emotion** and **drowsiness** detection has been implemented using **transfer learning** to create a holistic workplace wellness app. This uses **MediaPipe** to detect posture and industry-standard **RULA**, **REBA** algorithms to detect imperfect posture. Leveraged **AWS Cognito** to implement authentication and to encrypt face recognition data. **AWS DynamoDB** has been utilised to handle emotion, drowsiness and posture data to generate workplace satisfaction report.

In order to examine and reduce risks related to the ergonomics of a person's home workstation, this project intends to create an assessment tool. It guarantees that everyone can evaluate their posture and receive assistance in correcting it in order to avoid developing musculoskeletal disorders. Based on industry-standard ergonomics assessment procedures like Rapid Upper Limb Assessment (RULA) and Rapid Entire Body Assessment, this programme determines whether a person's posture is unsafe or not using live webcam video of them (REBA).

## Emotion and Drowsiness analysis

This also includes emotion and drowsiness analysis, where the software tracks your emotions over the course of the run and produces a report on your mental condition. The drowsiness detection component assists in alerting your mobile device if it determines that you are fatigued or not.

## Privacy

To ensure privacy, only your face will be tracked because live face tracking is enabled in this location. The face tracking data is securely encrypted, and the key is kept in the AWS Cognitio entries. The user can switch to private mode if he does not want to see videos or receive live feeds. and Only the coordinate data is transferred to the cloud from the camera's edge device (local computer), not the camera's video stream, which is instead transformed into coordinates. 

## Features

- Authentication using AWS Cognito
- Posture Detection and Correction
- Emotion Tracking
- Drowsiness Detection
- Face Tracking for increased accuracy
- Telegram based notification service to warn users about bad posture and drowsiness
- Report generation based on accumulated data stored in DynamoDB


## Run Locally

Clone the project

```bash
  git clone https://github.com/DeepanNarayanaMoorthy/Ergonomics-and-Emotion-Monitoring.git
```

Go to the project directory

```bash
  cd Ergonomics-and-Emotion-Monitoring
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run MainApp.py
```


## Roadmap

- Procedure followed for Drowsiness Detection
    - The model we used is built with Keras using Convolutional Neural Networks (CNN).
    - Image was taken as input from a webcam feed.
    - Face was detected in the image to create a Region of Interest (ROI).
    - Eyes are detected from ROI and fed to the classifier.
    - Classifier categorized whether eyes are open or closed.
    - Score was Calculated to check whether the person is drowsy.

- Procedures followed for Emotion Tracking
    - Created a list of every emotion that might exist in our dataset, along with a colour that corresponds to each emotion.
    - Established a technique for constructing the emotion classifier architecture.
    - It is given the input shape and the dataset's total number of classes.
    - This dataset's data was organised into columns for emotion, pixels, and usage in a CSV file.
    - The labels on all the photographs were one-hot encoded, and all the images were converted to NumPy arrays.
    - Each subset of data was loaded along with the saved model.
    - The cv2.VideoCapture() object was created in order to retrieve the webcam feed's frames.
    - For a face detector, the Haar Cascades were developed.
    - Each frame in the webcam stream was iterated over, with the loop ending only when there were no more frames to read.
    - The frames were resized to be 380 pixels wide (the height will be computed automatically to preserve the aspect ratio).
    - The input frames were changed to black and white since Haar Cascades only operate on grayscale images.
    - A face detector was then used on it.
    - checked to see whether any detections existed, then retrieved the one with the biggest area.
    - The emotions from the region of interest (roi) related to the detected face were retrieved.
    - The plot of emotion distribution was made.
    - The emotion that was displayed on the observed face was plotted.

## File Descriptions
- **Combined.py**: Posture, emotion, drowsiness related functions
- **angle_calc.py**: Function to derive angles from MediaPipe posture data
- **cognitio_auth.py**: AWS DynamoDB and Cognito functions
- **face_train.py**: Function to create face recognition data
- **multipage.py**: StreamLit Class to implement multiple page application

## Authors

- [Deepan Narayanamoorthy](https://github.com/DeepanNarayanaMoorthy)
- [Shankar Narayanan D](https://github.com/dshankar4)

## Tech Stack

**Client:** Streamlit

**Server:** Python, Tensorflow, OpenCV
