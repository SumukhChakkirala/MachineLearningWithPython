import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt

def detect_and_decode_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    barcodes = decode(gray)

    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        print("Barcode Data:", barcode_data)





image = cv2.imread("C:/Users/Sumukh/Downloads/image2.png")

detect_and_decode_barcode(image)