import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
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


model = load_model('Face-Recognition-Attendence-System-SQL\model.h5')

cascade = cv2.CascadeClassifier('Face-Recognition-Attendence-System-SQL\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

isOk = cv2.imread('Face-Recognition-Attendence-System-SQL\OK.png')
isOk = cv2.resize(isOk,(100,100))
notOk = cv2.imread('Face-Recognition-Attendence-System-SQL\OkNot.png')
notOk = cv2.resize(notOk,(100,100))
Checking = cv2.imread('Face-Recognition-Attendence-System-SQL\Checking.png')
Checking = cv2.resize(Checking,(100,100))

names = ['Aaron Taylor Johnson', 'Abigail Breslin', 'Adam Sandler', 'Adrianne Palicki', 'Alan Arkin', 'Alec Baldwin', 'Alexis Thorpe', 'Amanda Seyfried', 'Amy Adams', 'Andrew Garfield', 'Angelina Jolie', 'Anjelica Huston', 'Anna Kendrick', 'Anna Paquin', 'Annasophia Robb', 'Anthony Hopkins', 'Barbra Streisand', 'Ben Affleck', 'Ben Kingsley', 'Ben Stiller', 'Benedict Cumberbatch', 'Bette Midler', 'Betty White', 'Bill Murray', 'Brad Pitt', 'Bradley Cooper', 'Brenda Fricker', 'Bruce Willis', 'Bryan Cranston', 'Buster Keaton', 'Cameron Diaz', 'Carey Mulligan', 'Carol Burnett', 'Cary Grant', 'Cate Blanchett', 'Catherine Zeta Jones', 'Channing Tatum', 'Charlie Hunnam', 'Charlize Theron', 'Cher', 'Chloe Grace Moretz', 'Chris Cooper', 'Chris Evans', 'Chris Hemsworth', 'Christian Bale', 'Christina Ricci', 'Christopher Lee', 'Christopher Plummer', 'Christopher Walken', 'Cloris Leachman', 'Colin Farrell', 'Cuba Gooding Jr', 'Dakota Fanning', 'Daniel Craig', 'Daniel Radcliffe', 'Daryl Hannah', 'Dave Franco', 'David Niven', 'David Strathairn', 'Debbie Reynolds', 'Deborah Kerr', 'Debra Winger', 'Denzel Washington', 'Diane Keaton', 'Diane Lane', 'Don Cheadle', 'Donna Reed', 'Dustin Hoffman', 'Dwayne Johnson', 'Ed Harris', 'Elisabeth Moss', 'Eliza Dushku', 'Elizabeth Taylor', 'Ellen Burstyn', 'Ellen Page', 'Emily Watson', 'Emma Roberts', 'Emma Stone', 'Emma Thompson', 'Emma Watson', 'Ethan Hawke', 'Ethel Barrymore', 'F Murray Abraham', 'Faye Dunaway', 'Forest Whitaker', 'Frank Langella', 'Geena Davis', 'Gemma Arterton', 'Gene Hackman', 'Gene Wilder', 'Geoffrey Rush', 'George Chakiris', 'George Clooney', 'Gerard Butler', 'Gillian Anderson', 'Glenn Close', 'Goldie Hawn', 'Guy Pearce', 'Gwyneth Paltrow', 'Halle Berry', 'Hattie Mcdaniel', 'Helen Hunt', 'Henry Cavill', 'Hilary Swank', 'Holly Hunter', 'Hugh Grant', 'Hugh Jackman', 'Hugo Weaving', 'Ian Mckellen', 'Isla Fisher', 'Jake Gyllenhaal', 'James Franco', 'Jamie Foxx', 'Jamie Lee Curtis', 'Janet Gaynor', 'Jared Leto', 'Jason Bateman', 'Jason Segel', 'Jason Statham', 'Javier Bardem', 'Jean Dujardin', 'Jeff Bridges', 'Jeff Goldblum', 'Jennifer Aniston', 'Jennifer Connelly', 'Jennifer Garner', 'Jennifer Hudson', 'Jennifer Lawrence', 'Jeremy Renner', 'Jesse Eisenberg', 'Jessica Alba', 'Jessica Chastain', 'Jessica Lange', 'Jim Broadbent', 'Jim Carrey', 'Joaquin Phoenix', 'Jodie Foster', 'John Cusack', 'John Travolta', 'Johnny Depp', 'Jon Voight', 'Jonah Hill', 'Joseph Gordon Levitt', 'Jude Law', 'Judi Dench', 'Julia Louis Dreyfus', 'Julia Roberts', 'Julianne Moore', 'Julie Walters', 'June Squibb', 'Kaley Cuoco', 'Karl Urban', 'Kate Hudson', 'Kate Winslet', 'Katherine Heigl', 'Kathleen Turner', 'Kathy Bates', 'Katie Holmes', 'Keanu Reeves', 'Keira Knightley', 'Kevin Costner', 'Kevin Kline', 'Kim Basinger', 'Kirsten Dunst', 'Kristen Bell', 'Kristen Stewart', 'Kristen Wiig', 'Laura Linney', 'Leighton Meester', 'Lena Headey', 'Leonardo Dicaprio', 'Liam Hemsworth', 'Liam Neeson', 'Liv Tyler', 'Logan Lerman', 'Lupita Nyongo', 'Maggie Gyllenhaal', 'Marcia Gay Harden', 'Marisa Tomei', 'Mark Ruffalo', 'Mark Wahlberg', 'Marlee Matlin', 'Mary Steenburgen', 'Meg Ryan', 'Mel Gibson', 'Melissa Leo', 'Melissa Mccarthy', 'Mercedes Ruehl', 'Meryl Streep', 'Mia Wasikowska', 'Michael Caine', 'Michael Douglas', 'Michelle Pfeiffer', 'Michelle Williams', 'Monique', 'Neve Campbell', 'Nicholas Hoult', 'Nicolas Cage', 'Nicole Kidman', 'Octavia Spencer', 'Olivia Thirlby', 'Olivia Wilde', 'Owen Wilson', 'Patrick Swayze', 'Paul Giamatti', 'Paul Rudd', 'Paul Walker', 'Peter Dinklage', 'Queen Latifah', 'Rachel Mcadams', 'Rebel Wilson', 'Reese Witherspoon', 'Rene Russo', 'Robert De Niro', 'Robin Wright', 'Rooney Mara', 'Ryan Gosling', 'Ryan Reynolds', 'Sally Field', 'Sandra Bullock', 'Saoirse Ronan', 'Sean Penn', 'Selena Gomez', 'Seth Rogen', 'Shah Rukh Khan', 'Shailene Woodley', 'Shia Labeouf', 'Sidney Poitier', 'Sigourney Weaver', 'Sissy Spacek', 'Steve Carell', 'Steve Martin', 'Susan Sarandon', 'Sylvester Stallone', 'Taylor Kitsch', 'Terrence Howard', 'Thora Birch', 'Tilda Swinton', 'Tim Robbins', 'Timothy Hutton', 'Tina Fey', 'Tom Cruise', 'Tom Hanks', 'Tom Hardy', 'Tom Hiddleston', 'Tommy Lee Jones', 'Tyler Perry', 'Uma Thurman', 'Vera Farmiga', 'Viggo Mortensen', 'Vin Diesel', 'Will Ferrell', 'Will Smith', 'Winona Ryder', 'Woody Allen', 'Zach Galifianakis', 'Zooey Deschanel']

count = 0

while True:

    ret,frame = cap.read()

    frame = cv2.resize(frame,(640,480))

    num_face = cascade.detectMultiScale(image=frame,scaleFactor=1.3,minNeighbors=20)

    frame[370:470,270:370] = Checking

    if len(num_face) != 0:

        for x,y,w,h in num_face:

            count += 1

            draw_border(frame,(x,y),(x+w,y+h),(0,255,0),2,10,15)

            if count == 70: 
                face = frame[y:y+h,x:x+w]
                face = cv2.resize(face,(64,64))
                face = img_to_array(face)
                face = np.expand_dims(face,axis=0)
                face = preprocess_input(face)
                prediction = model.predict(face)
                prediction = prediction[0]
                prediction = np.where(prediction == 1)
                prediction = prediction[0]
                if prediction:
                    result = prediction[0]
                    name = names[result]
                    funQuery = FunctionsQuery(name)
                    funQuery.create_database()
                    funQuery.insert()
                    frame[370:470,270:370] = isOk
                    count = 0
                    time.sleep(1)
                else:
                    frame[370:470,270:370] = notOk
                    count = 0



    else:
        frame[370:470,270:370] = notOk
        count = 0


    cv2.imshow('Face',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()