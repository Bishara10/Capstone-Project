from django.shortcuts import redirect
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import HttpResponse
from django.template import loader
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from .models import signatures_collection, train_data_collection
import cv2
import numpy as np
import librosa 
import os
import parselmouth
import soundfile as sf
from parselmouth.praat import call
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from bson.binary import Binary


### MODEL ###

eye_tracking_input_shape = (3,)  # 3 values per sample: mean, median, std
reaction_time_input_shape_1 = (3,)
reaction_time_input_shape_2 = (3,)
audio_input_shape = (5,)  # 5 features: Fundamental Frequency, Jitter, Shimmer, Harmonics-to-noise ratio, Voice quality index
questionnaire_input_shape = (20,)  # 20 questions in total

# Eye tracking model
# eye_tracking_input = Input(shape=eye_tracking_input_shape, name='eye_tracking_input')
# x_eye = Dense(64, activation='relu')(eye_tracking_input)

# # Reaction time models
# reaction_time_input_1 = Input(shape=reaction_time_input_shape_1, name='reaction_time_input_1')
# reaction_time_input_2 = Input(shape=reaction_time_input_shape_2, name='reaction_time_input_2')
# x_reaction_time_1 = Dense(64, activation='relu')(reaction_time_input_1)
# x_reaction_time_2 = Dense(64, activation='relu')(reaction_time_input_2)

# # Audio recording model
# audio_input = Input(shape=audio_input_shape, name='audio_input')
# x_audio = Dense(64, activation='relu')(audio_input)
# x_audio = Dense(32, activation='relu')(x_audio)

# # Questionnaire model
# questionnaire_input = Input(shape=questionnaire_input_shape, name='questionnaire_input')
# x_questionnaire = Dense(64, activation='relu')(questionnaire_input)
# x_questionnaire = Dense(32, activation='relu')(x_questionnaire)

# # Concatenate all models
# concatenated = Concatenate()([x_eye, x_reaction_time_1, x_reaction_time_2, x_audio, x_questionnaire])
# x = Dense(64, activation='relu')(concatenated)
# x = Dense(32, activation='relu')(x)
# output = Dense(4, activation='softmax', name='output')(x)

# # Create model
# model_pred = Model(inputs=[eye_tracking_input, reaction_time_input_1, reaction_time_input_2, audio_input, questionnaire_input], outputs=output)

# # Compile model
# model_pred.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# weight_path = Path('ADHD/static/weights/model_weights (1).h5')

# # To load the weights later
# model_pred.load_weights(weight_path)
# model_pred.summary()

model_pred = Sequential()

# Input layer
input_shape = (3 + 5 + 20 + 3 + 3, 1)  # Combined input features
model_pred.add(Input(shape=input_shape))

# Conv1D layer with Batch Normalization and Dropout
model_pred.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
model_pred.add(BatchNormalization())
model_pred.add(MaxPooling1D(pool_size=2))
model_pred.add(Dropout(0.3))

# Additional Conv1D layer for deeper feature extraction
model_pred.add(Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
model_pred.add(BatchNormalization())
model_pred.add(MaxPooling1D(pool_size=2))
model_pred.add(Dropout(0.3))

# Flatten layer
model_pred.add(Flatten())

# Fully connected layers with Dropout and Regularization
model_pred.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model_pred.add(Dropout(0.4))
model_pred.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model_pred.add(Dropout(0.4))

# Output layer
model_pred.add(Dense(4, activation='softmax'))

# Compile the model
model_pred.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model_pred.summary()

weight_path = Path('ADHD/static/weights/93model15.weights.h5')
model_pred.load_weights(weight_path)

# Create your views here

# A function that loads the conesnt form page
def consentForm(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('consent-form-page.html')
    return HttpResponse(template.render())


# A function that saves the signature and full name from the consent form, it saves the data to a local folder
@csrf_exempt
def saveSignature(request):
    
    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    if request.method == 'POST' and request.FILES.get('signature'):
        signature = request.FILES['signature']
        full_name = request.POST['full-name']
        full_name = full_name.replace(" ","-")

        #Save signature to database
        name_and_session = str(full_name) + "_" + str(session_id)
        save_signature_toDB(full_name=name_and_session, signature=signature)

        signature_path = Path('ADHD/signatures/' + str(full_name) + "_" + str(session_id)+ "_signature.png" )
        with open(signature_path, 'wb+') as destination:
            for chunk in signature.chunks():
                destination.write(chunk)


    template = loader.get_template('consent-form-page.html')
    request.session['progress'] = 'step1'
    return HttpResponse(template.render())

# A function that loads the instruction page
def instructions(request):

    if request.session.get('progress') != 'step1' and request.session.get('progress') != 'step2':
        return redirect('/consent/')

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('instructions.html')
    request.session['progress'] = 'step2'
    return HttpResponse(template.render())

# A function that loads the eye-test page
def eyeTest(request):

    if request.session.get('progress') != 'step2':
        return redirect('/instructions')

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('eye-test.html')
    return HttpResponse(template.render())

# A function that loads the vocal-test page
def vocalTest(request):

    if request.session.get('progress') != 'step3':
        return redirect('/ADHD/')


    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('vocal-test.html')
    return HttpResponse(template.render())

# A function that loads the questionnaire page
def questionnaire(request):
    
    if request.session.get('progress') != 'step4':
        return redirect('/vocal-test')

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('questionnaire-styled.html')
    return HttpResponse(template.render())


# A function that loads the landing page
def landingPage(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    request.session['progress'] = 'step0'
    template = loader.get_template('nodus.html')
    return HttpResponse(template.render())

# A function that loads the frequently asked questions page
def FAQ(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    request.session['progress'] = 'step0'
    template = loader.get_template('FAQ.html')
    return HttpResponse(template.render())

# A function that loads the processing page (does nothing)
@csrf_exempt
def processing(request):

    if request.session.get('progress') != 'step5' and request.session.get('progress') != 'step5.5':
        return redirect('/questionnaire/')
    elif request.session.get('progress') != 'step5.5':
        request.session['progress'] = 'step5.5'
    else:
        request.session['progress'] = 'step6'

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('processing.html')
    return HttpResponse(template.render())

# A function that deletes the temporary files created by the user's tests
def clean(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    eye = False
    vocal = False
    question = False
    reaction = False
    initial = False

    save_path_initial = Path('ADHD/temporary_files/' + str(session_id)+ "_initial-video.webm" )
    save_path_eye = Path('ADHD/temporary_files/' + str(session_id)+ "_recorded-video.webm" )
    save_path_vocal = Path('ADHD/temporary_files/' + str(session_id)+ "_audio-recording.mp3" )
    save_path_questionnaire = Path('ADHD/temporary_files/' + str(session_id)+ "_questionnaire.txt" )
    save_path_reaction = Path('ADHD/temporary_files/'  + str(session_id)+ "_reaction-time-arrays.txt")
    try:    
        os.remove(save_path_eye)
        eye = True
    except:
        pass

    try:    
        os.remove(save_path_vocal)
        vocal = True
    except:
        pass

    try:    
        os.remove(save_path_questionnaire)
        question = True
    except:
        pass

    try:
        os.remove(save_path_reaction)
        reaction = True
    except:
        pass

    try:
        os.remove(save_path_initial)
        initial = True
    except: 
        pass
    
    ret = "Files deleted: "
    if eye:
        ret += "eye, "
    if vocal:
        ret += "vocal, "
    if question:
        ret += "question, "
    if reaction:
        ret += "reaction, "
    if initial:
        ret += "initial"
    
    return JsonResponse({'message': str(ret)}, status=200)  

# A function that analyzes the test data, runs the Deep Learning model and loads the results page
@csrf_exempt
def results(request):

    if request.session.get('progress') != 'step6' and request.session.get('progress') != 'step6.5':
        return redirect('/questionnaire/')
    elif request.session.get('progress') != 'step6.5':
        request.session['progress'] = 'step6.5'
    else:
        request.session['progress'] = 'step1'

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    template = loader.get_template('test-results.html')

    # Feed data to DL model and retrieve the probabilities
    ip = request.META['REMOTE_ADDR'] # Extract IP to conduct analysis on files with this IP
    save_path_eye = Path('ADHD/temporary_files/' + str(session_id)+ "_recorded-video.webm" )
    save_path_vocal = Path('ADHD/temporary_files/' + str(session_id)+ "_audio-recording.mp3" )
    save_path_questionnaire = Path('ADHD/temporary_files/' + str(session_id)+ "_questionnaire.txt" )
    save_path_reaction = Path('ADHD/temporary_files/'  + str(session_id)+ "_reaction-time-arrays.txt")
    
    questionnaire = ''
    reaction = ""
    reaction_time_base = ''
    reaction_time_dist = ''


    try:
        voice_analysis = analyze_voice(save_path_vocal)
    except:
        return JsonResponse({'message': 'Error - could not analyze voice'}, status=201)
    
    try:
        eye_analysis = eye_test_analysis(save_path_eye)
    except:
        return JsonResponse({'message': 'Error - could not analyze eye movements'}, status=201)
    
    try:
        with open(save_path_questionnaire, 'r') as f:
            questionnaire += f.read()
    except:
        return JsonResponse({'message': 'Error - could not access questionnaire answers'}, status=201)
    
    try:
        with open(save_path_reaction, 'r') as f:
            reaction += f.read()
    except:
        return JsonResponse({'message': 'Error - could not access reaction time data'}, status=201)
    
    reaction_time_base, reaction_time_dist = reaction.split("\n")

    reaction_time_base = reaction_time_base[1:]
    reaction_time_base = reaction_time_base[:-1]
    base_array = reaction_time_base.split(',')
    base_array = [int(i) for i in base_array]

    reaction_time_dist = reaction_time_dist[1:]
    reaction_time_dist = reaction_time_dist[:-1]
    dist_array = reaction_time_dist.split(',')
    dist_array = [int(i) for i in dist_array]

    base, dist = calculate_mean_median_std(base_array), calculate_mean_median_std(dist_array)

    
    if (not questionnaire.split(',')[-1].isnumeric()):
        data_file_path = Path('ADHD/temporary_files/data.txt')
        with open(data_file_path,'a') as f:

            eye = eye_analysis
            vocal = voice_analysis
            questions = tuple(map(int, questionnaire.split(',')[:-1]))
            reactions_base = base
            reactions_dist = dist
            string = ""

            for i in eye:
                string += str(i) + ", "
            for i in vocal:
                string += str(i) + ", "
            for i in questions:
                string += str(i) + ", "
            for i in reactions_base:
                string += str(i) + ", "
            for i in range(len(reactions_dist)):
                string += str(reactions_dist[i]) + ", "
            label = getLabel(questionnaire)   
            string += "Train Mode: " + label
            save_test_results_toDB(label, string)

            
    else:
        questions = tuple(map(int, questionnaire.split(',')))


    percentages = model(eye_analysis, base, dist, voice_analysis, questions)
    tips = get_tips(percentages)
    context = {
        'percentages': percentages,
        'tips': tips,
        'eye_analysis': eye_analysis,
        'vocal_analysis': voice_analysis,
        'questionnaire': questionnaire,
        'reaction_time': reaction.split('\n')
    } 
    return HttpResponse(template.render(context=context))

# A helper function that calculates what the user's subtype is
def getLabel(questionnaire):
    inat = 0
    hyp = 0

    index = 1
    for answer_str in questionnaire.split(",")[:-1]:
        if index > 18:
            break
        answer = int(answer_str)
        if index <= 9:
            if answer == 2 or answer == 3:
                inat += 1
        else:
            if answer == 2 or answer == 3:
                hyp += 1

        index += 1

    if hyp >= 6 and inat >= 6:

        return "Combined"
    
    elif hyp >= 6:
        return "Hyperactivity"
    
    elif inat >= 6:
        return "Inattentive"
    
    else:
        return "No ADHD"


# A helper function that yields tips according to the user's subtype
def get_tips(percentages):
    
    combined = "Balance your day with both physical activities and quieter tasks, using visual schedules and verbal reminders to stay organized. Break tasks into smaller steps and check in with yourself or ask for help to ensure you stay on track. Engage in structured physical activities and plan movement breaks to manage hyperactivity. Use positive reinforcement and behavior plans with specific goals and rewards to encourage good behavior and task completion. Collaborate with teachers and involve your family in supporting you. Enjoy consistent feedback and support from those around you to manage both inattentive and hyperactive symptoms effectively."
    hyperactive = "Incorporate plenty of physical activities and movement breaks into your day to help manage energy. Engage in structured activities with clear rules and fewer choices to maintain focus. Play games that teach self-control, and watch how adults model patience. Use a behavior chart to track and reward positive behaviors like waiting your turn and sitting still during meals. Channel your energy into sports, crafts, and organized play. Remember to take short breaks during tasks that require sitting still. Enjoy consistent and immediate rewards for good behavior to understand what is expected of you."
    inattentive = "Create a consistent daily routine and use visual aids to remember what tasks come next. Break instructions into small, manageable steps, and repeat them back to ensure you understand. Set up a quiet, clutter-free area for focused activities like homework. Celebrate small achievements with immediate praise or rewards to stay motivated. Regularly check in with yourself or ask for help to stay on track. Simplify your environment by minimizing distractions and keeping only necessary items around. Set small, achievable goals and enjoy the satisfaction of completing each step."
    no_adhd = "While ADHD is often a neurodevelopmental disorder present from early childhood, maintaining a healthy lifestyle can support overall brain health and well-being. Establish a consistent daily routine with balanced activities to reduce stress and improve focus. Prioritize a nutritious diet rich in fruits, vegetables, lean proteins, and whole grains, and avoid excessive sugar and processed foods. Ensure you get regular physical exercise, which can enhance brain function and reduce anxiety. Maintain good sleep hygiene by having a consistent bedtime and creating a relaxing sleep environment. Engage in activities that challenge your mind, such as puzzles, reading, and learning new skills. Limit screen time and take regular breaks from electronic devices to prevent overstimulation. Finally, practice mindfulness and relaxation techniques like deep breathing or meditation to manage stress and improve concentration."

    tips_array = [combined, hyperactive, inattentive, no_adhd]

    return tips_array[np.argmax(percentages)]

# A function that saves the video created by the eye-test
@csrf_exempt
def upload_video(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    if request.method == 'POST' and request.FILES.get('initial-video'):
        video = request.FILES['initial-video']
        save_path = Path('ADHD/temporary_files/' + str(session_id)+ "_" + video.name )
        with open(save_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)
        
        res = initial_video_check(save_path)
        if(res):
            return JsonResponse({'message': 'Confiuguration complete'}, status=200)
        else:
            return JsonResponse({'message': 'No eyes detected, try sitting closer to the screen, brighten or dampen the room'}, status=201)
        
    elif request.method == 'POST' and request.FILES.get('recorded-video'):
        video = request.FILES['recorded-video']
        base_reaction_time = request.POST['reactionTimeBase']
        dis_reaction_time = request.POST['reactionTimeDistractors']
        save_path = Path('ADHD/temporary_files/' + str(session_id)+ "_" + video.name )
        with open(save_path, 'wb+') as destination:
            for chunk in video.chunks():
                destination.write(chunk)
        save_path_arrays = Path('ADHD/temporary_files/'  + str(session_id)+ "_reaction-time-arrays.txt")
        with open(save_path_arrays, 'w+') as destination:
            destination.write(base_reaction_time)
            destination.write('\n')
            destination.write(dis_reaction_time)

        res = True
        if(res):
            request.session['progress'] = 'step3'
            return JsonResponse({'message': str(res)}, status=200)
        else:
            return JsonResponse({'message': 'No eyes detected, try sitting closer to the screen, brighten or dampen the room'}, status=201)

    return JsonResponse({'error': 'Invalid request'}, status=400)

# A function that loads the initial video created by the eye-test configuration and checks if the eyes are detected
def initial_video_check(url):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(url)
    detected = 0
    undetected = 0

    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces) != 0:
            for (x,y,w,h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_color)
                if len(eyes) != 0:
                    detected += 1
                else:
                    undetected += 1
        else:
            undetected += 1

    cap.release()
    return detected > undetected*1.5 and detected > 40


# A function that loads the eye-test video and analyzes eye movement data
def eye_test_analysis(url):


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    initial_coords_left = [-1,-1]
    initial_coords_right = [-1,-1]
    face_center = [-1,-1]
    eye_coordinates = [-1,-1]

    cap = cv2.VideoCapture(url)
    left_eye_coordinates_array = []
    right_eye_coordinates_array = []
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x,y,w,h) in faces:
            face_center = [int(h/2),int(w/2)]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                roi = roi_color[ey:ey+eh, ex:ex+ew]
                # Left eye
                if ex < face_center[1]:
                    if initial_coords_left == [-1,-1]:
                        initial_coords_left = [ey + int(eh/2), ex + int(ew/2)]
                # Right eye
                else:
                    if initial_coords_right == [-1,-1]:
                        initial_coords_right = [ey + int(eh/2), ex + int(ew/2)]
                eye_coordinates = [ey + int(eh/2), ex + int(ew/2)]
        try:
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
            _, threshold = cv2.threshold(gray_roi, 115, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
                break
            # Left eye
            if eye_coordinates[1] < face_center[1]:
                if len(left_eye_coordinates_array) == 0:
                    left_eye_coordinates_array.append(((initial_coords_left[1] - eye_coordinates[1]), (initial_coords_left[0] - eye_coordinates[0])))
                elif left_eye_coordinates_array[-1] != ((initial_coords_left[1] - eye_coordinates[1]), (initial_coords_left[0] - eye_coordinates[0])):
                    left_eye_coordinates_array.append(((initial_coords_left[1] - eye_coordinates[1]), (initial_coords_left[0] - eye_coordinates[0])))
            else:
                if len(right_eye_coordinates_array) == 0:
                    right_eye_coordinates_array.append(((initial_coords_left[1] - eye_coordinates[1]), (initial_coords_left[0] - eye_coordinates[0])))
                elif right_eye_coordinates_array[-1] != ((initial_coords_left[1] - eye_coordinates[1]), (initial_coords_left[0] - eye_coordinates[0])):
                    right_eye_coordinates_array.append(((initial_coords_left[1] - eye_coordinates[1]), (initial_coords_left[0] - eye_coordinates[0])))
        except:
            pass

    cap.release()
    difference_right = calculate_difference(right_eye_coordinates_array)
    difference_left = calculate_difference(left_eye_coordinates_array)
    mean_right, median_right, std_right = calculate_mean_median_std(difference_right)
    mean_left, median_left, std_left = calculate_mean_median_std(difference_left)
    mean = (mean_right + mean_left)/2
    median = (median_right + median_left)/2
    std = (std_right + std_left)/2
    return mean, median, std

# A helper function that calculates the squared difference between eye positions for each step
def calculate_difference(eye_position_array):
    difference = []
    for i in range(1,len(eye_position_array)):
        difference.append(distance_squared(eye_position_array[i-1], eye_position_array[i]))
    return difference

# A helper function that calculates the squared difference between 2 points
def distance_squared(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2)

# A helper function that calculates the mean, median and standard deviation of an array\list of numbers
def calculate_mean_median_std(eye_movement_distance_array):
    data = []
    for coord in eye_movement_distance_array:
        data.append(coord)
    return np.mean(data), np.median(data), np.std(data)

# A function that saves the audio file created by the vocal-test
@csrf_exempt
def upload_voice(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    if request.method == 'POST' and request.FILES.get('audio-recording'):
        audio = request.FILES['audio-recording']
        save_path = Path('ADHD/temporary_files/' + str(session_id)+ "_" + audio.name )
        with open(save_path, 'wb+') as destination:
            for chunk in audio.chunks():
                destination.write(chunk)
        y, sr = librosa.load(save_path)
        sf.write(save_path, y, sr)
        request.session['progress'] = 'step4'
    return JsonResponse({'message': 'Uploaded video successfully'}, status=200)

# A function that loads the vocal-test audio file and analyzes the vocal characterisitcs of the user
def analyze_voice(url):

    #y, sr = librosa.load(url, sr=None)
    
    # Extract Fundamental Frequency (F0)
    f0 = extract_fundamental_frequency(str(url))
    
    # Extract Jitter, Shimmer, and HNR
    jitter, shimmer, hnr = extract_jitter_shimmer_hnr(str(url))
    
    # Voice Quality Index (VQI) - example of composite metric
    vqi = (f0 / 100) + jitter + shimmer + (hnr / 10)
    return f0, jitter, shimmer, hnr, vqi

# A function that saves the questionnaire answers created by the questionnaire page
@csrf_exempt
def upload_answers(request):

    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    if request.method == 'POST' and request.POST.get('questionnaire'): 
        answers = request.POST['questionnaire']
        save_path_arrays = Path('ADHD/temporary_files/' + str(session_id)+ "_questionnaire.txt")
        with open(save_path_arrays, 'w+') as destination:
            destination.write(answers)
        request.session['progress'] = 'step5'
    return JsonResponse({'message': "Questionnaire answers uploaded successfully"}, status=200)

# A helper function that extracts the Fundamental Frequency of the user's voice (F0)
def extract_fundamental_frequency(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    return mean_f0

# A helper function that extracts the various vocal characteristics of the user's voice, specifically Jitter, Shimmer and Harmonics-to-Noise Ratio
def extract_jitter_shimmer_hnr(audio_path):
    snd = parselmouth.Sound(audio_path)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    
    # Jitter (local)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # Shimmer (local)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # Harmonics-to-Noise Ratio (HNR)
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.01, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return jitter, shimmer, hnr

# A function that receives the tests' results as tuples and turns them into lists, then into np.arrays and runs the model on these parameters to get the predicted results
# def model(eye, react1, react2, vocal, questions):

#     eye_list = list(eye)
#     react1_list = list(react1)
#     react2_list = list(react2)
#     vocal_list = list(vocal)
#     questions_list = list(questions)

#     eye_np = np.ndarray([eye_list])
#     react1_np = np.ndarray([react1_list])
#     react2_np = np.ndarray([react2_list])
#     vocal_np = np.ndarray([vocal_list])
#     questions_np = np.ndarray([questions_list])

#     # prediction = model_pred.predict([eye_np, react1_np, react2_np, vocal_np, questions_np])
#     prediction = model_pred.predict(np.ndarray(eye_np, vocal_np, questions_np, react1_np, react2_np))

#     return prediction[0]

def model(eye, react1, react2, vocal, questions):

    tup = []

    for item in eye:
      tup.append(item)
    for item in vocal:
      tup.append(item)
    for item in questions:
      tup.append(item)
    for item in react1:
      tup.append(item)
    for item in react2:
      tup.append(item)

    ndarray_of_arrays = np.array([np.array([x]) for x in tup])

    final = np.array(ndarray_of_arrays)
    final = np.expand_dims(final, axis=0)
    prediction = model_pred.predict(final)

    return prediction[0]


# ========================================================== Database methods ==========================================================

def save_signature_toDB(full_name, signature):
    if isinstance(signature, InMemoryUploadedFile):
        # Read the file content
        file_data = signature.read()
        
        # Convert the file content to BSON Binary
        binary_signature = Binary(file_data)
        
        # Create the record to be saved
        record = {
            "full_name": full_name,
            "signature": binary_signature
        }
        
        # Insert the record into MongoDB
        signatures_collection.insert_one(record)
        return HttpResponse("New Person is added.")
    else:
        return HttpResponse("Invalid file type.", status=400)


def save_test_results_toDB(label, data):
    record = {
        label: data
    }

    train_data_collection.insert_one(record)

def get_all_signatures(request):
    signatures = signatures_collection.find()
    return signatures

# def dummy_data(request):
#     string = "135.34005764054643, 16.5, 612.1943038200075, 166.3011130613908, 0.03299776024567752, 0.16355241628927905, 5.392012914699131, 2.3987625986187773, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 397.2857142857143, 373.0, 65.09710172550868, 403.8888888888889, 408.0, 55.61863086022014, Train model: No ADHD"
#     label = "No ADHD"
#     save_test_results_toDB(label,string)
