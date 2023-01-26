from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import datetime, time
import os, sys
import numpy as np
from time import perf_counter
from threading import Thread
from animations.main import MU_effect
from animations.trial import tiktok_animation 
from animations.segmentation import Segmentation
from animations.doctorStrange import doctor_strange

app=Flask(__name__, static_url_path='/static/', template_folder='static/templates')
app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = 'Upload'
global global_frame, video_camera, rec, cap
rec, cap = 1, 1
video_camera = None
global_frame = None

def generate_frames_1():
    camera=cv2.VideoCapture(0)
    i = 0
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = tiktok_animation(frame, 1, i)
        try:
            i+=1
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

def record_generate_frames_1():
    camera=cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Upload/Anima1.avi', fourcc, 20.0, (1080, 500))
    i = 0
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            print('Not sucess')
            break
        else:
            frame = tiktok_animation(frame, 1, i)
            out.write(frame)
        try:
            i+=1
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass
    out.release()

def generate_frames_2():
    camera=cv2.VideoCapture(0)
    camera.get(cv2.CAP_PROP_BUFFERSIZE)    
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = MU_effect(frame)
        try:
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass
    
def record_generate_frames_2():
    camera=cv2.VideoCapture(0)
    camera.get(cv2.CAP_PROP_BUFFERSIZE)    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Upload/Anima2.avi', fourcc, 20.0, (640, 480))
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = MU_effect(frame)
            out.write(frame)
        try:
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass
    out.release()

def generate_frames_3():
    camera=cv2.VideoCapture(0)
    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale_fact = 1
    segment_count = fps*3
    segment_height = int(height*scale_fact/segment_count)
    frames = []
    t1 = perf_counter()
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            if scale_fact != 1:
                frame = cv2.resize(frame, (int(frame.shape[1]*scale_fact), int(frame.shape[0]*scale_fact)))
            frames.append(frame)
            if len(frames) >= segment_count:    
                segments = []
                for i,frame in enumerate(frames):
                    segments.append(frame[i*segment_height:(i+1)*segment_height])
                frame = np.concatenate(segments, axis=0)

                frames.pop(0)
                t2 = perf_counter()
                delay = int(1000/fps - (t2-t1)*1000)
                delay = delay if delay >1 else 1
                t1 = perf_counter()
            try:
                ret, buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

def record_generate_frames_3():
    camera=cv2.VideoCapture(0)
    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale_fact = 1
    segment_count = fps*3
    segment_height = int(height*scale_fact/segment_count)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Upload/Anima3.avi', fourcc, 20.0, (640, 450))
    frames = []
    temp = None
    t1 = perf_counter()
    count = 0
    while True:
        count+=1
        success,frame=camera.read()
        X = frame
        if not success:
            break
        else:
            if scale_fact != 1:
                frame = cv2.resize(frame, (int(frame.shape[1]*scale_fact), int(frame.shape[0]*scale_fact)))
            frames.append(frame)
            # print('scale fact: ', scale_fact, ' segment height: ', segment_height)
            if len(frames) >= segment_count:    
                segments = []
                for i,frame in enumerate(frames):
                    X = frame[i*segment_height:(i+1)*segment_height]
                    print(X)
                    segments.append(X)
                frame = np.concatenate(segments, axis=0)
                cv2.imwrite(f'video/pic_{count}.jpg', frame)
                out.write(frame)
                frames.pop(0)
                t2 = perf_counter()
                delay = int(1000/fps - (t2-t1)*1000)
                delay = delay if delay >1 else 1
                t1 = perf_counter()
            try:
                ret, buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
    out.release()

def generate_frames_4():
    camera=cv2.VideoCapture(0)
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.GaussianBlur(frame, (23, 23), 0, 0)
            img_blend = cv2.divide(frame, img_blur, scale=200)
        try:
            ret, buffer=cv2.imencode('.jpg',img_blend)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img_blend + b'\r\n')
        except Exception as e:
                pass

def record_generate_frames_4():
    camera=cv2.VideoCapture(0)
    camera.get(cv2.CAP_PROP_BUFFERSIZE)    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Upload/Anima4.avi', fourcc, 40.0, (640, 480))
    while True:
        success,frame= camera.read()
        if not success:
            break
        else:
            frame = doctor_strange(frame)
            out.write(frame)
        try:
            ret, buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
                pass
    out.release()

def generate_frames_rmBG():
    camera=cv2.VideoCapture(0)
    imgBg = cv2.imread("Photo/9.jpg")
    seg = Segmentation()
    while True:
        # read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame, imgBg = cv2.resize(frame, (640, 480)), cv2.resize(imgBg, (640, 480))
            frame = seg.removeBG(frame, imgBg, threshold=0.55)
        try:
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
                pass

def record_generate_frames_rmBG():
    camera=cv2.VideoCapture(0)
    imgBg = cv2.imread("Photo/9.jpg")
    seg = Segmentation()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Upload/rmBG.avi', fourcc, 40.0, (640, 480))
    while True:
        success,frame= camera.read()
        if not success:
            break
        else:
            frame, imgBg = cv2.resize(frame, (640, 480)), cv2.resize(imgBg, (640, 480))
            frame = seg.removeBG(frame, imgBg, threshold=0.55)
            out.write(frame)
        try:
            ret, buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
                pass
    out.release()

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    # Appending app path to upload folder path within app root folder
    uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, path=filename)

@app.route('/animation_1', methods=['POST','GET'])
def Animation1():
    global rec, cap
    if request.method == 'POST':
        if request.form.get('cap') == "Turn on/off Webcam":
            cap *= -1
            return render_template('Animation1.html', cap=cap)
        elif request.form.get('rec') == "Recording":
            rec *= -1
            return render_template('Animation1.html', rec=rec)
    else:
        return render_template('Animation1.html')


@app.route('/animation_2', methods=['POST','GET'])
def Animation2():
    global rec, cap
    if request.method == 'POST':
        if request.form.get('cap') == "Turn on/off Webcam":
            cap *= -1
            return render_template('Animation2.html', cap=cap)
        elif request.form.get('rec') == "Recording":
            rec *= -1
            return render_template('Animation2.html', rec=rec)
    else:
        return render_template('Animation2.html')


@app.route('/animation_3', methods=['POST','GET'])
def Animation3():
    global rec, cap
    if request.method == 'POST':
        if request.form.get('cap') == "Turn on/off Webcam":
            cap *= -1
            return render_template('Animation3.html', cap=cap)
        elif request.form.get('rec') == "Recording":
            rec *= -1
            return render_template('Animation3.html', rec=rec)
    else:
        return render_template('Animation3.html')

@app.route('/animation_4', methods=['POST','GET'])
def Animation4():
    global rec, cap
    if request.method == 'POST':
        if request.form.get('cap') == "Turn on/off Webcam":
            cap *= -1
            return render_template('Animation4.html', cap=cap)
        elif request.form.get('rec') == "Recording":
            rec *= -1
            return render_template('Animation4.html', rec=rec)
    else:
        return render_template('Animation4.html')

@app.route('/Background_removal', methods=['POST','GET'])
def Background_removal():
    global rec, cap
    if request.method == 'POST':
        if request.form.get('cap') == "Turn on/off Webcam":
            cap *= -1
            return render_template('Background-removal.html', cap=cap)
        elif request.form.get('rec') == "Recording":
            rec *= -1
            return render_template('Background-removal.html', rec=rec)
    else:
        return render_template('Background-removal.html')


@app.route('/file/video_animation_1')
def video_1():
    return Response(generate_frames_1(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/file/video_animation_2')
def video_2():
    return Response(generate_frames_2(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/file/video_animation_3')
def video_3():
    return Response(generate_frames_3(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/file/video_animation_4')
def video_4():
    return Response(generate_frames_4(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/file/BG_removal')
def video_BG_removal():
    return Response(generate_frames_rmBG(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recording/video_animation_1')
def record_video_1():
    return Response(record_generate_frames_1(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/recording/video_animation_2')
def record_video_2():
    return Response(record_generate_frames_2(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/recording/video_animation_3')
def record_video_3():
    return Response(record_generate_frames_3(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/recording/video_animation_4')
def record_video_4():
    return Response(record_generate_frames_4(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/recording/BG_removal')
def record_video_BG_removal():
    return Response(record_generate_frames_rmBG(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
    
 