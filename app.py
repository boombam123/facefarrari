import tkinter as tk
from tkinter import filedialog
import os
import cv2
import sys
import cv2 #opencv itself
import common #some useful opencv functions
import numpy as np # matrix manipulations
import imutils
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
import requests
from flask import Flask, jsonify, flash, send_file, request,url_for,redirect,abort

app = Flask(__name__)

@app.route('/pan')
def pan_matchinger():
    return send_file('pan_matchinger.jpg', mimetype='image/gif')

@app.route('/adhar')
def adhar_matchinger():
    return send_file('adhar_matchinger.jpg', mimetype='image/gif') 

    
# @app.route('/message')
# def rr():
#     return 'Message Prnted'
    
    
@app.route('/')
def temp():
        return '''
    <!doctype html>
<title>Face Matching</title>
<h2>Select file(s) to upload</h2>
<h3>Steps to follow:\n1)Upload Aadhar\n2)Upload PAN\n3)Upload haarcascade file\n4) Capture photo of user<h3/>
    <form method="post" action="/">
        <input type="submit" value="Click Here to Start Matching!" name="action1"/>
    </form>
    '''   
    

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        if request.form.get('action1') == 'Click Here to Start Matching!':
            root=tk.Tk()
            root.withdraw()
            filePath1=filedialog.askopenfilename()
            filePath2=filedialog.askopenfilename()
            filePath3=filedialog.askopenfilename()
            rootF=os.getcwd()
            print(rootF)    
            image_extraction_adhar(filePath1, filePath3)
            image_extraction_pan(filePath2, filePath3)

            import cv2
            cam = cv2.VideoCapture(0)
            cv2.namedWindow("test")
            img_counter = 0
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("test", frame)

                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1
            cam.release()
            cv2.destroyAllWindows()
            revognizer_with_adhar(rootF)
            revognizer_with_pan(rootF)
            return adhar_matchinger()
        # elif  request.form.get('action2') == 'VALUE2':
        #     return redirect('localhost:5000/message')
        else:
            abort(401)
    


def image_extraction_adhar(filePath1, filePath3):
      #the following are to do with this interactive notebook code
      #matplotlib inline 
      from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks
      import pylab # this allows you to control figure size 
      
      image_adhar=cv2.imread(filePath1)   
      i=0
      while True:
        # Get user supplied values
        imagePath = sys.argv[0]

        # Load the Haar Cascade 
        #cascPath =r'E:\yash\haarcascade_frontalface_default.xml'
        cascPath=filePath3
        # Create the Haar Cascade
        faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the Image
        #image = cv2.imread('/content/inverted coloured.jpeg')

        # Convert to Gray-Scale
        gray = cv2.cvtColor(image_adhar, cv2.COLOR_BGR2GRAY)

        # Detect Faces in the Image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25)
        )
        #i=0
        if(len(faces)>0):
          #print(i)
          #base_image = cv2.imread('/content/colored  adar.jpeg')
          imag=imutils.rotate(image_adhar,angle=i)
          grey = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
          plt.imshow(cv2.cvtColor(imag, cv2.COLOR_BGR2RGB))

          # test_image = cv2.imread('/content/inverted coloured.jpeg')
          # im=imutils.rotate(test_image,angle=i)
          face_cascade = cv2.CascadeClassifier(filePath3)
          faces = face_cascade.detectMultiScale(grey, 1.3, 5)
          for (x,y,w,h) in faces:
              cv2.rectangle(imag,(x,y),(x+w,y+h),(255,0,0),2)
              face_crop=image_adhar[y:y+h,x:x+w]
              cv2.imwrite('facecrop_adhar.jpg',face_crop)
              
          plt.imshow(cv2.cvtColor(imag, cv2.COLOR_BGR2RGB))
          break
        else:
          # print(i)
          # i+=1
          # image=imutils.rotate(image,angle=i)
          # #cv2.imwrite('/contents/inverted coloured.jpeg',img)
          # #i+=1
          print("Check the orientation")
          break
      print("Found {0} faces!".format(len(faces)))



def image_extraction_pan(filePath2, filePath3):

      #the following are to do with this interactive notebook code
      #matplotlib inline 
      # this allows you to control figure size 
      image_pan=cv2.imread(filePath2) 
      i=0
      while True:
        # Get user supplied values
        imagePath = sys.argv[0]

        # Load the Haar Cascade 
        cascPath = filePath3

        # Create the Haar Cascade
        faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the Image
        #image = cv2.imread('/content/inverted coloured.jpeg')

        # Convert to Gray-Scale
        gray = cv2.cvtColor(image_pan, cv2.COLOR_BGR2GRAY)

        # Detect Faces in the Image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25)
        )
        #i=0
        if(len(faces)>0):
          #print(i)
          #base_image = cv2.imread('/content/colored  adar.jpeg')
          imag=imutils.rotate(image_pan,angle=i)
          grey = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
          plt.imshow(cv2.cvtColor(imag, cv2.COLOR_BGR2RGB))

          # test_image = cv2.imread('/content/inverted coloured.jpeg')
          # im=imutils.rotate(test_image,angle=i)
          face_cascade = cv2.CascadeClassifier(filePath3)
          faces = face_cascade.detectMultiScale(grey, 1.3, 5)
          for (x,y,w,h) in faces:
              cv2.rectangle(imag,(x,y),(x+w,y+h),(255,0,0),2)
              face_crop=image_pan[y:y+h,x:x+w]
              cv2.imwrite('facecrop_paaaaan.jpg',face_crop)
              
          plt.imshow(cv2.cvtColor(imag, cv2.COLOR_BGR2RGB))
          break
        else:
          # print(i)
          # i+=1
          # image=imutils.rotate(image,angle=i)
          # #cv2.imwrite('/contents/inverted coloured.jpeg',img)
          # #i+=1
          print("Check the orientation")
          break
      print("Found {0} faces!".format(len(faces)))


def revognizer_with_adhar(rootF):
                from PIL import Image, ImageDraw
                from IPython.display import display

                # The program we will be finding faces on the example below
                pil_im = Image.open(r'{0}\opencv_frame_0.png'.format(rootF))
                display(pil_im)

                import face_recognition
                import numpy as np
                from PIL import Image, ImageDraw
                from IPython.display import display

                # This is an example of running face recognition on a single image
                # and drawing a box around each person that was identified.

                # Load a sample picture and learn how to recognize it.
                yash_image = face_recognition.load_image_file(r"{0}\facecrop_adhar.jpg".format(rootF))
                yash_face_encoding = face_recognition.face_encodings(yash_image)[0]

                # # Load a second sample picture and learn how to recognize it.
                # sachin_image = face_recognition.load_image_file("/content/image13.jpeg")
                # sachin_face_encoding = face_recognition.face_encodings(sachin_image)[0]

                # Create arrays of known face encodings and their names
                known_face_encodings = [
                    yash_face_encoding
                    #,
                    # sachin_face_encoding
                ]
                known_face_names = [
                    "Matching with adhar"
                    # ,
                    # "Sachin"
                ]
                print('Learned encoding for', len(known_face_encodings), 'images.')


                # Load an image with an unknown face
                unknown_image = face_recognition.load_image_file(r"{0}\opencv_frame_0.png".format(rootF))

                # Find all the faces and face encodings in the unknown image
                face_locations = face_recognition.face_locations(unknown_image)
                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

                # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
                # See http://pillow.readthedocs.io/ for more about PIL/Pillow
                pil_image = Image.fromarray(unknown_image)
                # Create a Pillow ImageDraw Draw instance to draw with
                draw = ImageDraw.Draw(pil_image)

                # Loop through each face found in the unknown image
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unmatched with adhar"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    # Draw a box around the face using the Pillow module
                    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                    # Draw a label with a name below the face
                    text_width, text_height = draw.textsize(name)
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


                # Remove the drawing library from memory as per the Pillow docs
                del draw

                # Display the resulting image
                display(pil_image)
                pil_image = pil_image.save("adhar_matchinger.jpg")
               




def revognizer_with_pan(rootF):
                from PIL import Image, ImageDraw
                from IPython.display import display

                # The program we will be finding faces on the example below
                pil_im = Image.open(r'{0}\opencv_frame_0.png'.format(rootF))
                display(pil_im)

                import face_recognition
                import numpy as np
                from PIL import Image, ImageDraw
                from IPython.display import display

                # This is an example of running face recognition on a single image
                # and drawing a box around each person that was identified.

                # Load a sample picture and learn how to recognize it.
                yash_image = face_recognition.load_image_file(r"{0}\facecrop_paaaaan.jpg".format(rootF))
                yash_face_encoding = face_recognition.face_encodings(yash_image)[0]

                # # Load a second sample picture and learn how to recognize it.
                # sachin_image = face_recognition.load_image_file("/content/image13.jpeg")
                # sachin_face_encoding = face_recognition.face_encodings(sachin_image)[0]

                # Create arrays of known face encodings and their names
                known_face_encodings = [
                    yash_face_encoding
                    #,
                    # sachin_face_encoding
                ]
                known_face_names = [
                    "Matching with pan"
                    # ,
                    # "Sachin"
                ]
                print('Learned encoding for', len(known_face_encodings), 'images.')


                # Load an image with an unknown face
                unknown_image = face_recognition.load_image_file(r"{0}\opencv_frame_0.png".format(rootF))

                # Find all the faces and face encodings in the unknown image
                face_locations = face_recognition.face_locations(unknown_image)
                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

                # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
                # See http://pillow.readthedocs.io/ for more about PIL/Pillow
                pil_image = Image.fromarray(unknown_image)
                # Create a Pillow ImageDraw Draw instance to draw with
                draw = ImageDraw.Draw(pil_image)

                # Loop through each face found in the unknown image
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unmatched with pan"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    # Draw a box around the face using the Pillow module
                    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

                    # Draw a label with a name below the face
                    text_width, text_height = draw.textsize(name)
                    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
                    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


                # Remove the drawing library from memory as per the Pillow docs
                del draw

                # Display the resulting image
                display(pil_image)
                pil_image = pil_image.save("pan_matchinger.jpg")
               



if __name__=="__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)
