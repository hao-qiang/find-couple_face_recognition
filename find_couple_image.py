import face_recognition
import cv2
import numpy as np
import os
import sys


img_path = sys.argv[1]
couple_dir = "./data/couple/"
single_dir = "./data/single/"

# extract couple's facial features
couple_imgs_names = os.listdir(couple_dir)
if len(couple_imgs_names)==2:
    if '_' in couple_imgs_names[0]:
        gf_img_name = couple_imgs_names[0]
        gf_name = gf_img_name.split('.')[0][:-1]
        bf_img_name = couple_imgs_names[1]
        bf_name = bf_img_name.split('.')[0]
    else:
        gf_img_name = couple_imgs_names[1]
        gf_name = gf_img_name.split('.')[0][:-1]
        bf_img_name = couple_imgs_names[0]
        bf_name = bf_img_name.split('.')[0]
else:
    raise Exception("You should put a pair of pictures in folder \'couple\'!")

known_face_names = []
known_face_encodings = []
known_face_names.append(gf_name)
gf_img = face_recognition.load_image_file(os.path.join(couple_dir,gf_img_name))
known_face_encodings.append(face_recognition.face_encodings(gf_img)[0])
known_face_names.append(bf_name)
bf_img = face_recognition.load_image_file(os.path.join(couple_dir,bf_img_name))
known_face_encodings.append(face_recognition.face_encodings(bf_img)[0])

# extract singles' facial features
single_imgs_names = os.listdir(single_dir)
if len(single_imgs_names) != 0:
    for img_name in single_imgs_names:
        known_face_names.append(img_name.split('.')[0])
        single_img = face_recognition.load_image_file(os.path.join(single_dir,img_name))
        known_face_encodings.append(face_recognition.face_encodings(single_img)[0])


process_this_frame = True


while True:
    frame = cv2.imread(img_path)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if name == bf_name:
            top_x_bf = (right + left) // 2
            top_y_bf = top
            
        if name == gf_name:
            top_x_gf = (right + left) // 2
            top_y_gf = top
            # make up
            face_landmarks_list = face_recognition.face_landmarks(frame[top-10:bottom+10,left-10:right+10,:][:,:,::-1])
            if face_landmarks_list!=[]:
                triangle = np.asarray([face_landmarks_list[0]['left_eyebrow']], np.int32)+(left-10,top-10)
                cv2.fillConvexPoly(frame, triangle, (39, 54, 68, 128), lineType=4)
                triangle = np.asarray([face_landmarks_list[0]['right_eyebrow']], np.int32)+(left-10,top-10)
                cv2.fillConvexPoly(frame, triangle, (39, 54, 68, 128), lineType=4)
                triangle = np.asarray([face_landmarks_list[0]['top_lip']], np.int32)+(left-10,top-10)
                cv2.fillConvexPoly(frame, triangle, (0, 0, 150, 128), lineType=4)
                triangle = np.asarray([face_landmarks_list[0]['bottom_lip']], np.int32)+(left-10,top-10)
                cv2.fillConvexPoly(frame, triangle, (0, 0, 150, 128), lineType=4)
                triangle = np.asarray([face_landmarks_list[0]['left_eye']], np.int32)+(left-10,top-10)
                cv2.polylines(frame, triangle, True, (0, 0, 0, 255), thickness=2, lineType=4)
                triangle = np.asarray([face_landmarks_list[0]['right_eye']], np.int32)+(left-10,top-10)
                cv2.polylines(frame, triangle, True, (0, 0, 0, 255), thickness=2, lineType=4)
            
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    if (bf_name in face_names) and (gf_name in face_names):
        target_x = (top_x_bf + top_x_gf) // 2
        target_y = min(top_y_bf, top_y_gf) - 50
        cv2.line(frame,(top_x_bf,top_y_bf),(target_x, target_y),(0,0,255), 3)
        cv2.line(frame,(top_x_gf,top_y_gf),(target_x, target_y),(0,0,255), 3)
        cv2.ellipse(frame,(target_x-7, target_y),(30,20),60,0,360,(0,0,255),-1)
        cv2.ellipse(frame,(target_x+7, target_y),(30,20),-60,0,360,(0,0,255),-1)
        
    # Display the resulting image
    cv2.imshow('Image', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cv2.destroyAllWindows()