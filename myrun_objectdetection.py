#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: www.github.com/GustavZ
"""
import numpy as np
import tensorflow as tf
import os
from rod.helper import FPS, WebcamVideoStream, SessionWorker, conv_detect2track, conv_track2detect, vis_detection, Timer
from rod.model import Model
from rod.config import Config
from rod.utils import ops as utils_ops
import Image
import threading
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time

vs = None

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Tracker
        if self.config.USE_TRACKER:
            import sys
            sys.path.append(os.getcwd()+'/stuff/kcf')
            import KCF
            tracker = KCF.kcftracker(False, True, False, False)
            tracker_counter = 0
            track = False

        print("> Building Graph")
        # tf Session Config
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth=True
        detection_graph = self.model.detection_graph
        category_index = self.model.category_index
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<img src="http://127.0.0.1:8082/cam.mjpg"/>')
            self.wfile.write('</body></html>')
            return
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            with detection_graph.as_default():
                with tf.Session(graph=detection_graph,config=tf_config) as sess:
                    # start Videostream
                    vs = WebcamVideoStream(self.config.VIDEO_INPUT,self.config.WIDTH,self.config.HEIGHT).start()
                    # Define Input and Ouput tensors
                    tensor_dict = self.model.get_tensordict(['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks'])
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Mask Transformations
                    if 'detection_masks' in tensor_dict:
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                detection_masks, detection_boxes, vs.real_height, vs.real_width)
                        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                    if self.config.SPLIT_MODEL:
                        score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                        expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                        score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                        expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                        # Threading
                        score = self.model.score
                        expand = self.model.expand
                        gpu_worker = SessionWorker("GPU",detection_graph,tf_config)
                        cpu_worker = SessionWorker("CPU",detection_graph,tf_config)
                        gpu_opts = [score_out, expand_out]
                        cpu_opts = [tensor_dict['detection_boxes'], tensor_dict['detection_scores'], tensor_dict['detection_classes'], tensor_dict['num_detections']]
                        gpu_counter = 0
                        cpu_counter = 0

                    fps = FPS(self.config.FPS_INTERVAL).start()
                    print('> Starting Detection')

                    while vs.isActive():
                        # Detection
                        if not (self.config.USE_TRACKER and track):
                            if self.config.SPLIT_MODEL:
                                # split model in seperate gpu and cpu session threads
                                masks = None # No Mask Detection possible yet
                                if gpu_worker.is_sess_empty():
                                    # read video frame, expand dimensions and convert to rgb
                                    frame = vs.read()
                                    # put new queue
                                    gpu_feeds = {image_tensor: vs.expanded()}
                                    if self.config.VISUALIZE:
                                        gpu_extras = frame # for visualization frame
                                    else:
                                        gpu_extras = None
                                    gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)
                                g = gpu_worker.get_result_queue()
                                if g is None:
                                    # gpu thread has no output queue. ok skip, let's check cpu thread.
                                    gpu_counter += 1
                                else:
                                    # gpu thread has output queue.
                                    gpu_counter = 0
                                    score,expand,frame = g["results"][0],g["results"][1],g["extras"]

                                    if cpu_worker.is_sess_empty():
                                        # When cpu thread has no next queue, put new queue.
                                        # else, drop gpu queue.
                                        cpu_feeds = {score_in: score, expand_in: expand}
                                        cpu_extras = frame
                                        cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)
                                c = cpu_worker.get_result_queue()
                                if c is None:
                                    # cpu thread has no output queue. ok, nothing to do. continue
                                    cpu_counter += 1
                                    continue # If CPU RESULT has not been set yet, no fps update
                                else:
                                    cpu_counter = 0
                                    boxes, scores, classes, num, frame = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
                            else:
                                # default session
                                frame = vs.read()
                                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: vs.expanded()})
                                num = output_dict['num_detections'][0]
                                classes = output_dict['detection_classes'][0]
                                boxes = output_dict['detection_boxes'][0]
                                scores = output_dict['detection_scores'][0]
                                if 'detection_masks' in output_dict:
                                    masks = output_dict['detection_masks'][0]
                                else:
                                    masks = None

                            # reformat detection
                            num = int(num)
                            boxes = np.squeeze(boxes)
                            classes = np.squeeze(classes).astype(np.uint8)
                            scores = np.squeeze(scores)

                            # Visualization
                            vis = vis_detection(frame, boxes, classes, scores, masks, category_index, fps.fps_local(),
                                                self.config.VISUALIZE, self.config.DET_INTERVAL, self.config.DET_TH, self.config.MAX_FRAMES, fps._glob_numFrames)
                            if not vis:
                                break
                            # output to stream
                            jpg = Image.fromarray(vis)
                            tmpFile = StringIO.StringIO()
                            jpg.save(tmpFile, 'JPEG')
                            self.wfile.write("--jpgboundary")
                            self.send_header('Content-type', 'image/jpeg')
                            self.send_header('Content-length', str(tmpFile.len))
                            self.end_headers()
                            jpg.save(self.wfile, 'JPEG')
                            time.sleep(0.001)
                            # end output
                            # Activate Tracker
                            if self.config.USE_TRACKER and num <= self.config.NUM_TRACKERS:
                                tracker_frame = frame
                                track = True
                                first_track = True

                        # Tracking
                        else:
                            frame = vs.read()
                            if first_track:
                                trackers = []
                                tracker_boxes = boxes
                                for box in boxes[~np.all(boxes == 0, axis=1)]:
                                        tracker.init(conv_detect2track(box,vs.real_width, vs.real_height), tracker_frame)
                                        trackers.append(tracker)
                                first_track = False

                            for idx,tracker in enumerate(trackers):
                                tracker_box = tracker.update(frame)
                                tracker_boxes[idx,:] = conv_track2detect(tracker_box, vs.real_width, vs.real_height)
                            vis = vis_detection(frame, tracker_boxes, classes, scores, masks, category_index, fps.fps_local(),
                                                self.config.VISUALIZE, self.config.DET_INTERVAL, self.config.DET_TH, self.config.MAX_FRAMES, fps._glob_numFrames)
                            if not vis:
                                break

                            tracker_counter += 1
                            #tracker_frame = frame
                            if tracker_counter >= self.config.TRACKER_FRAMES:
                                track = False
                                tracker_counter = 0

                        fps.update()

            # End everything
            vs.stop()
            fps.stop()
            if self.config.SPLIT_MODEL:
                gpu_worker.stop()
                cpu_worker.stop()


if __name__ == '__main__':
    config = Config()
    model = Model('od',config.OD_MODEL_NAME,config.OD_MODEL_PATH,config.LABEL_PATH,
                config.NUM_CLASSES,config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
    # detection(model, config)
    global vs
    global vis
    global frame
    try:
        server = ThreadedHTTPServer(('localhost', 8082), CamHandler)
        server.model = model
        server.config = config
        print("server started")
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()

