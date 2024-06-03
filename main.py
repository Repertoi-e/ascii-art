import numpy as np
import pygame
from pygame.surfarray import pixels_red, pixels_green, pixels_blue
import cv2

from pdf2image import convert_from_path
   
import io
import sys
from PIL import Image, ImageDraw, ImageFont

import concurrent.futures
import time

def normalize(arr):
    "Maps the values in the array to a [0, 1] range"
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


default = list(reversed(['M', 'W', 'Q', 'B', 'E', 'R', 'N', '@', 'H', 'q', 'p', 'g', 'K', 'A', '#', 'm', 'b', '8', '0', 'd', 'X', 'D', 'G', 'F', 'P', 'e', 'h', 'U', '9', '6', 'k', 'Z', '%', 'S', '4', 'O', 'x', 'y', 'T', '5', 'w', 'f', 'a', 'V', 's',
'2', 'L', '$', 'Y', '&', 'n', '3', 'C', 'J', 'u', 'o', 'z', 'I', 'j', 'v', 'c', 'r', 't', 'l', 'i', '1', '=', '?', '7', '>', '<', ']', '[', '(', ')', '+', '*', ';', '}', '{', ':', '/', '\\', '!', '|', '_', ',', '^', '-', '~', '.', ' ']))

boxes = [' ', '░', '▒', '▓', '█']

characters = boxes

def rgb_to_grayscale(img):
    "Gamma corrected, OpenCV's RGB2GRAY does the same."
    r = pixels_red(img)
    g = pixels_green(img)
    b = pixels_blue(img)
    return (0.30 * r + 0.59 * g + 0.11 * b).T

def rgb_to_grayscale_numpy(img):
    "Gamma corrected, OpenCV's RGB2GRAY does the same."
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    return (0.30 * r + 0.59 * g + 0.11 * b).T

def convolution2d(image, kernel, bias=0, skip_x=1, skip_y=1):
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (img_height - kernel_height) // skip_y + 1
    output_width = (img_width - kernel_width) // skip_x + 1

    view_shape = (output_height, output_width, kernel_height, kernel_width)
    strides = (image.strides[0] * skip_y, image.strides[1] * skip_x, image.strides[0], image.strides[1])
    sub_matrices = np.lib.stride_tricks.as_strided(image, shape=view_shape, strides=strides)

    result = np.tensordot(sub_matrices, kernel, axes=((2, 3), (0, 1))) + bias

    return result

def get_frame_stringified(y, kernel_size=(3, 5)):
    kernel_width, kernel_height = kernel_size

    avg_kernel = np.ones((kernel_height, kernel_width)) / (kernel_width * kernel_height)
    convolved = normalize(convolution2d(y, kernel=avg_kernel, skip_x=kernel_width, skip_y=kernel_height))

    indices = np.abs(convolved[:, :, None] - np.linspace(0, 1, len(characters))).argmin(axis=2)
    char_image = np.vectorize(lambda x: characters[x])(indices)

    return '\n'.join(''.join(row) for row in char_image)


def get_every_10th_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            # Convert frame to RGB format (optional, depending on your needs)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_rgb)
        
        frame_count += 1

    cap.release()
    return frames

def do_video(video_path):
    frames = get_every_10th_frame(video_path)
    for i, frame in enumerate(frames):
        print(get_frame_stringified(frame))
        time.sleep(0.3)

# do_video(video_path='dynamics.mp4')

def do_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(get_frame_stringified(frame))

                time.sleep(0.3)
        except KeyboardInterrupt:
            print("Capture interrupted by user.")
        finally:
            # Release the webcam
            cap.release()
            print("Webcam released.")

def capture_console_output(func, *args, **kwargs):
    # Redirect stdout to capture print statements
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        func(*args, **kwargs)
    finally:
        sys.stdout = sys.__stdout__  # Restore original stdout
    return buffer.getvalue()

def calculate_image_size(text, font):
    l, t, r, b = font.getbbox('A')

    lines = text.split('\n')
    max_width = 0
    total_height = 0
    for line in lines:
        max_width = max(max_width, len(line) * (r - l))
        total_height += b   
    return max_width, total_height

def render_text_to_image(text, font_path='/System/Library/Fonts/Menlo.ttc', font_size=5):
    font = ImageFont.truetype(font_path, font_size)

    image_size = calculate_image_size(text, font)
    print(image_size)
    image = Image.new("RGBA", image_size, color=(30, 30, 30))

    d_usr = ImageDraw.Draw(image)
    d_usr = d_usr.text((0, 0), text, fill=(220, 220, 220), font=font, spacing=0)
    return image

def read_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return [np.array(image) for image in images]

def process_page(page_index, page, kernel_size, font_size):
    def to_display_function():  
        print(get_frame_stringified(normalize(rgb_to_grayscale_numpy(page).T), kernel_size=kernel_size))

    captured_text = capture_console_output(to_display_function)
    image = render_text_to_image(captured_text, font_size=font_size)
    image_path = f'out/console_output_{page_index}.png'
    image.save(image_path)
    print(f"Done img: {page_index}")
    return image_path

def convert_pdf(pdf_path, kernel_size, font_size=5):
    pages = read_pdf_to_images(pdf_path)

    image_paths = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_page, i, pages[i], kernel_size, font_size) for i in range(len(pages))]
        for future in concurrent.futures.as_completed(futures):
            image_paths.append(future.result())

    def images_to_pdf(image_paths, output_pdf_path):
        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        images[0].save(output_pdf_path, save_all=True, append_images=images[1:])

    output_pdf_path = 'output.pdf'
    images_to_pdf(sorted(image_paths), output_pdf_path)


def convert_img(img_path, kernel_size, font_size=5):
    img = pygame.image.load(img_path)

    def to_display_function():  
        print(get_frame_stringified(normalize(rgb_to_grayscale(img)), kernel_size=kernel_size))

    captured_text = capture_console_output(to_display_function)
    image = render_text_to_image(captured_text,font_size=font_size)
    image.save(f'console_output.png')
    image.show()
    print(f"Done img: {img_path}")

if __name__ == '__main__':
    # convert_img('swarm_of_drones.jpg', font_size=10, kernel_size=(3, 5))
    convert_pdf('E02G_Wing_spar.pdf', kernel_size=(2, 4), font_size=5)
