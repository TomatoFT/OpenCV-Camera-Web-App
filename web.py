from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from time import perf_counter
from threading import Thread
from animations.main import *
from animations.trial import tiktok_animation

app=Flask(__name__, static_url_path='/static/', template_folder='static/templates')
app.static_folder = 'static'
camera=cv2.VideoCapture(0)

def generate_frames_1():
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

def generate_frames_2():
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

def generate_frames_3():
    fps = camera.get(cv2.CAP_PROP_FPS)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    scale_fact = 1
    segment_count = fps*3
    segment_height = int(height*scale_fact/segment_count)
    frames = []
    t1 = perf_counter()
    while True:
        ## read the camera frame
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
@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/animation_1')
def Animation1():
    return render_template('Animation1.html')
@app.route('/animation_2')
def Animation2():
    return render_template('Animation2.html')
@app.route('/animation_3')
def Animation3():
    return render_template('Animation3.html')


@app.route('/file/video_animation_1')
def video_1():
    return Response(generate_frames_1(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/file/video_animation_2')
def video_2():
    return Response(generate_frames_2(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/file/video_animation_3')
def video_3():
    return Response(generate_frames_3(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)