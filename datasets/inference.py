import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,font_scale=0.5, thickness=1):
                                                
    x, y = coordinates[:2]
    #cv2.rectangle(image_array, (x + 60, y + y_offset),(x + x_offset, y + y_offset-15), (255, 255, 255), cv2.FILLED)
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)




