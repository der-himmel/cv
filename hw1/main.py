import cv2
from tkinter import *
import platform
import subprocess
import sys
import re

# TODO: это в словарь превратить и обращаться по ключу, ключ - название определенной через get_os_info() ОС
VIDEO_CAP_BACKENDS = [
    cv2.CAP_ANY,
    getattr(cv2, "CAP_V4L2", 200),          # linux
    getattr(cv2, "CAP_DSHOW", 700),         # windows
    getattr(cv2, "CAP_MSMF", 1400),         # windows_alt
    getattr(cv2, "CAP_AVFOUNDATION", 1200), # mac
]
MAX_VIDEO_CAP_SOURCES_QUANTITY = 3

def get_os_info():
    os_info = platform.platform()
    if "linux" in os_info.lower():
        return "linux"
    if "windows" in os_info.lower():
        return "windows" 
    
def detect_video_sources():
    vidstream_sources = []
    os_info = get_os_info()
    for idx in range(MAX_VIDEO_CAP_SOURCES_QUANTITY):
        video_capture = None
        for backend in VIDEO_CAP_BACKENDS:
            video_capture = cv2.VideoCapture(idx, backend)
            pass
    
    
if __name__ == "__main__":
    main()
