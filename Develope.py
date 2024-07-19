import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from Attendance_System import FunctionsQuery
import time



def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

names = ['Adriana Lima',
 'Alex Lawther',
 'Alexandra Daddario',
 'Alvaro Morte',
 'Amanda Crew',
 'Andy Samberg',
 'Anne Hathaway',
 'Anthony Mackie',
 'Avril Lavigne',
 'Ben Affleck',
 'Bill Gates',
 'Bobby Morley',
 'Brenton Thwaites',
 'Brian J. Smith',
 'Brie Larson',
 'Chris Evans',
 'Chris Hemsworth',
 'Chris Pratt',
 'Christian Bale',
 'Cristiano Ronaldo',
 'Danielle Panabaker',
 'Dominic Purcell',
 'Dwayne Johnson',
 'Eliza Taylor',
 'Elizabeth Lail',
 'Emilia Clarke',
 'Emma Stone',
 'Emma Watson',
 'Gwyneth Paltrow',
 'Henry Cavil',
 'Hugh Jackman',
 'Inbar Lavi',
 'Irina Shayk',
 'Jake Mcdorman',
 'Jason Momoa',
 'Jennifer Lawrence',
 'Jeremy Renner',
 'Jessica Barden',
 'Jimmy Fallon',
 'Johnny Depp',
 'Josh Radnor',
 'Katharine Mcphee',
 'Katherine Langford',
 'Keanu Reeves',
 'Krysten Ritter',
 'Leonardo Dicaprio',
 'Lili Reinhart',
 'Lindsey Morgan',
 'Lionel Messi',
 'Logan Lerman',
 'Madelaine Petsch',
 'Maisie Williams',
 'Maria Pedraza',
 'Marie Avgeropoulos',
 'Mark Ruffalo',
 'Mark Zuckerberg',
 'Megan Fox',
 'Miley Cyrus',
 'Millie Bobby Brown',
 'Morena Baccarin',
 'Morgan Freeman',
 'Nadia Hilker',
 'Natalie Dormer',
 'Natalie Portman',
 'Neil Patrick Harris',
 'Pedro Alonso',
 'Penn Badgley',
 'Rami Malek',
 'Rebecca Ferguson',
 'Richard Harmon',
 'Rihanna',
 'Robert De Niro',
 'Robert Downey Jr',
 'Sarah Wayne Callies',
 'Selena Gomez',
 'Shakira Isabel Mebarak',
 'Sophie Turner',
 'Stephen Amell',
 'Taylor Swift',
 'Tom Cruise',
 'Tom Hardy',
 'Tom Hiddleston',
 'Tom Holland',
 'Tuppence Middleton',
 'Ursula Corbero',
 'Wentworth Miller',
 'Zac Efron',
 'Zendaya',
 'Zoe Saldana',
 'Alycia Dabnem Carey',
 'Amber Heard',
 'Barack Obama',
 'Barbara Palvin',
 'Camila Mendes',
 'Elizabeth Olsen',
 'Ellen Page',
 'Elon Musk',
 'Gal Gadot',
 'Grant Gustin',
 'Jeff Bezos',
 'Kiernen Shipka',
 'Margot Robbie',
 'Melissa Fumero',
 'Scarlett Johansson',
 'Tom Ellis']


model = load_model('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\model.h5')

cascade = cv2.CascadeClassifier('D:\Project-VSCode\Face-Recognition-RealTime-Attendance-System-SQL-main\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


while True:

    ret,frame = cap.read()

    frame = cv2.resize(frame,(640,640))

    num_face = cascade.detectMultiScale(image=frame,scaleFactor=1.3,minNeighbors=20)

    for x,y,w,h in num_face:

        draw_border(frame,(x,y),(x+w,y+h),(0,255,0),2,10,15)
    
        
        face = frame[y:y+h,x:x+w]
        face = cv2.resize(face,(160,160))
        face = img_to_array(face)
        face = np.expand_dims(face,axis=0) / 255
        prediction = model.predict(face)
        prediction = np.argmax(prediction)

        if prediction != 0:
            lable = names[prediction]
            cv2.putText(frame,lable,(x+10,y-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)
            # append names and time open and exit in database
            # funQuery = FunctionsQuery(lable)
            # funQuery.create_database()
            # funQuery.insert()
            # time.sleep(3)
        else:
            lable = 'Unknow'
            cv2.putText(frame,lable,(x+10,y-20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),1)

    cv2.imshow('Project',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()