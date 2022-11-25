# Reel Recommender System

# Import necessary Python Libraries
import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer       #ffpyplayer for playing audio
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Import Facial Emotion Recognition Model
model = load_model('Models/vgg16_v4.h5', compile = False)

# Preprocess Input Image and Make Prediction
image_dir = 'Dataset/test/angry/PrivateTest_3309033.jpg'

#load the image
my_image = load_img(image_dir, target_size=(48, 48))

#preprocess the image
my_image = img_to_array(my_image)
my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
my_image = preprocess_input(my_image)

#make the prediction
prediction = model.predict(my_image)
out = np.argmax(prediction)

# Define function to play video reel
def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()

# Video Reel Dictionary

# #####    
# video_path="Dataset/output/Happy/Cleaned/jonny-1.mp4"  
# PlayVideo(video_path)
# ####

##dictionary method
Happy= {'1':"Dataset/output/Happy/Cleaned/Babu-1.mp4",
        '2':"Dataset/output/Happy/Cleaned/Babu-2.mp4",
        '3':"Dataset/output/Happy/Cleaned/Babu-3.mp4",
        '4':"Dataset/output/Happy/Cleaned/Babu-4.mp4",
        '5':"Dataset/output/Happy/Cleaned/jonny-1.mp4",
        '6':"Dataset/output/Happy/Cleaned/raj-1.mp4",
        '7':"Dataset/output/Happy/Cleaned/vill-1.mp4",
        '8':"Dataset/output/Happy/Cleaned/Babu-5.mp4"
        }

Sad = {'1':"Dataset/output/Sad/Cleaned/Friends Moments that Will Make You Cry!.mp4 - VLC media player 2022-11-23 00-34-38.mp4",
       '2':"Dataset/output/Sad/Cleaned/Guptill_ Neesham_ _ India v New Zealand - Top 5 Moments _ ICC Cricket World Cup 2019 - YouTube — Mozilla Firefox 2022-11-23 00-41-16.mp4",
       '3':"Dataset/output/Sad/Cleaned/indian Army surgical strike Sad Song with victory & celebration song.mp4 - VLC media player 2022-11-23 00-25-49.mp4",
       '4':"Dataset/output/Sad/Cleaned/Parents Love ❤️😍 Heart touching speech by Paresh Rawal.mp4",
       '5':"Dataset/output/Sad/Cleaned/Try Not To Cry - February Amazon Prime Video.mp4 - VLC media player 2022-11-23 00-27-53.mp4",
       '6':"Dataset/output/Sad/Cleaned/Try Not To Cry - February Amazon Prime Video.mp4 - VLC media player 2022-11-23 00-29-14.mp4",
       '7':"Dataset/output/Energetic/Cleaned/Avengers End Game.mp4",
       '8':r"Dataset/output/Neutral/cleaned/Yesterday is a History, Master Oogway quote.mp4"
       }

Neutral = {'1':r"Dataset/output/Neutral/cleaned/10 CRAZY STORIES From India's History ft. Abhijit Chavda _ The Ranveer Show हिंदी 30 - YouTube — Mozilla Firefox 2022-11-22 20-17-35.mp4",
           '2':r"Dataset/output/Neutral/cleaned/10 CRAZY STORIES From India's History ft. Abhijit Chavda _ The Ranveer Show हिंदी 30 - YouTube — Mozilla Firefox 2022-11-22 20-19-04.mp4",
           '3':r"Dataset/output/Neutral/cleaned/One Minute Facts about India India facts in 1 Minute India.mp4",
           '4':r"Dataset/output/Neutral/cleaned/Tryst with Destiny _ Jawaharlal Nehru - YouTube — Mozilla Firefox 2022-11-22 19-54-38.mp4",
           '5':r"Dataset/output/Neutral/cleaned/What Happens In One Minute .mp4",
           '6':r"Dataset/output/Neutral/cleaned/Yesterday is a History, Master Oogway quote.mp4",
           '7':r"Dataset/output/Energetic/Cleaned/Dhoni.mp4",
           '8':"Dataset/output/Energetic/Cleaned/gabba.mp4"
           }

Energetic = {'1':"Dataset/output/Energetic/Cleaned/Avengers End Game.mp4",
             '2':"Dataset/output/Energetic/Cleaned/Conversation between Indira Gandhi and Rakesh Sharma.mp4",
             '3':"Dataset/output/Energetic/Cleaned/Dhoni.mp4",
             '4':"Dataset/output/Energetic/Cleaned/gabba.mp4",
             '5':"Dataset/output/Energetic/Cleaned/India’s spacecraft Mangalyaan reaches Mars.mp4",
             '6':"Dataset/output/Energetic/Cleaned/ronaldo.mp4",
             '7':"Dataset/output/Energetic/Cleaned/Thor arrives in wakanda!.mp4",
             '8':"Dataset/output/Energetic/Cleaned/Vande Mataram - A.R. Rahman_Maa Tujhe.mp4"}

# Knowledge base for reel recommendation

# 0-angry,1-disgust,2-fear,3-happy,4-neutral,5-sad,6-surprise
# Angry->Happy; Disgust->Happy;Fear->happy,energetic;Happy->happy,neutral,excited,sad; Neutral->Happy,neutral,excited; Sad-> Happy,excited; Surprise-> Excited, Happy
###Ifelse###
if out==0:
    video_path = Happy[str(random.randint(1, 8))]
    PlayVideo(video_path)
    
elif out==1:
    video_path = Happy[str(random.randint(1, 8))]
    PlayVideo(video_path)

elif out==2:
    a=random.choice([Happy,Energetic])
    video_path = a[str(random.randint(1, 8))]
    PlayVideo(video_path)

elif out==3:
    a = random.choice([Happy,Energetic,Neutral,Sad])
    video_path = a[str(random.randint(1, 8))]
    PlayVideo(video_path)

elif out==4:
    a = random.choice([Happy,Energetic,Neutral])
    video_path = a[str(random.randint(1, 8))]
    PlayVideo(video_path)

elif out==5:
    a=random.choice([Happy,Energetic])
    video_path = a[str(random.randint(1, 8))]
    PlayVideo(video_path)

else:
    a=random.choice([Happy,Energetic])
    video_path = a[str(random.randint(1, 8))]
    PlayVideo(video_path)


