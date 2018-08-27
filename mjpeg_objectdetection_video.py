#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: www.github.com/GustavZ
"""
from rod.model import ObjectDetectionModel
from rod.config import Config
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
vs = None

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        model_type = 'od'
        input_type = 'video'
        config = Config(model_type)
        model = ObjectDetectionModel(config).prepare_model(input_type)
        model.run(BaseHTTPRequestHandler)

if __name__ == '__main__':
    global vs
    global vis
    global frame
    try:
        server = ThreadedHTTPServer(('localhost', 8082), CamHandler)
        print("server started")
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()
