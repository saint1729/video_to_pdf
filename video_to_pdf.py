import os.path
import glob
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict


USERNAME, PDF_EXT = "saint1729", ".pdf"


def get_vtt_and_pdf_file_path(input_file_name, method=1):
    input_video_tokens = input_file_name.split("/")
    video_folder, file_name = "/".join(input_video_tokens[:-1]), input_video_tokens[-1]
    file_name_without_ext, ext = os.path.splitext(file_name)
    file_path_without_ext = video_folder + "/" + file_name_without_ext
    return file_path_without_ext + ".vtt", file_path_without_ext + str(method) + ".pdf"


def are_dissimilar_images2(curr_gray_image, prev_gray_image, bin_threshold=100, threshold=1000):
    ret, curr_thresh = cv2.threshold(curr_gray_image, bin_threshold, 255, cv2.THRESH_BINARY)
    ret, prev_thresh = cv2.threshold(prev_gray_image, bin_threshold, 255, cv2.THRESH_BINARY)
    diff = np.abs(curr_thresh.astype(np.int16) - prev_thresh.astype(np.int16))
    # cv2.imwrite("./debug/" + str(j) + "_diff.jpg", diff)
    return np.sum(diff == 255) > threshold


def get_full_image_with_text(input_image, texts):
    frame_image = Image.fromarray(input_image)
    dims = input_image.shape
    subtitle_image = Image.fromarray(np.zeros((50 + 50 * len(texts) + 50, dims[1], dims[2]), dtype=np.uint8))
    W, H = subtitle_image.size
    draw = ImageDraw.Draw(subtitle_image)
    my_font = ImageFont.truetype("Courier.ttc", 48)
    for idx, text in enumerate(texts):
        w, h = draw.textsize(text)
        draw.text(((W - w - 2000) / 2, 50 + idx * 50), text.center(80), font=my_font, fill="white")
    full_image = Image.fromarray(np.concatenate([frame_image, subtitle_image], axis=0))
    return full_image


def get_pdfs_path(input_file_name):
    input_video_tokens = input_file_name.split("/")
    video_folder = "/".join(input_video_tokens[:-1])
    return glob.glob(video_folder + "/*.pdf")


def generate_pdf_from_images(pdf_path, images):
    if len(images) > 1:
        images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
    elif len(images) == 1:
        images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True)


def convert_time_to_sec(time):
    tokens = time.split(":")
    tokens[2] = tokens[2].replace(",", ".")
    hh, mm, ss = float(tokens[0]), float(tokens[1]), float(tokens[2]) + 0.5
    return ss + mm * 60 + hh * 3600


def are_dissimilar_images(img1, img2, threshold):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # print(descriptors_1.shape)
    # print(descriptors_2.shape)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)

    return (len(matches) / min(descriptors_1.shape[0], descriptors_2.shape[0])) < threshold


def convert_to_pdf_1(input_video_link, subtitle_file, pdf_name, bin_threshold=100, white_pixel_count=1000):
    vid_cap = cv2.VideoCapture(input_video_link)
    f = open(subtitle_file)
    subtitle_lines = f.read().splitlines()

    cv_images, pil_images = [], []

    vid_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, prev_img = vid_cap.read()
    texts, j = [], 0

    for i, line in enumerate(subtitle_lines):
        if " --> " in line:
            times = line.split(" --> ")
            time_in_sec = convert_time_to_sec(times[0]) * 1000
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, time_in_sec)
            success, image = vid_cap.read()
            if success:
                current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                curr_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prev_gray_image = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

                if are_dissimilar_images2(curr_gray_image, prev_gray_image, bin_threshold, white_pixel_count)\
                        or (len(texts) > 15):
                    pil_images.append(get_full_image_with_text(prev_img, texts))
                    texts, prev_img = [], current_image
                texts.append(subtitle_lines[i + 1])
            else:
                print(f"failed in {line} for {subtitle_file}")

    pil_images.append(get_full_image_with_text(prev_img, texts))
    generate_pdf_from_images(pdf_name, pil_images)

    f.close()


def convert_to_pdf_2(input_video_link, subtitle_file, pdf_name, number_of_lines=10):
    vid_cap = cv2.VideoCapture(input_video_link)

    f = open(subtitle_file)
    subtitle_lines = f.read().splitlines()

    pil_images = []

    vid_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, current_image = vid_cap.read()

    texts, j = [], 0

    for i, line in enumerate(subtitle_lines):
        if " --> " in line:
            times = line.split(" --> ")
            time_in_sec = convert_time_to_sec(times[0]) * 1000
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, time_in_sec)
            success, image = vid_cap.read()
            if success:
                current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                j += 1

                if j % number_of_lines == 0:
                    pil_images.append(get_full_image_with_text(current_image, texts))
                    texts = []
                texts.append(subtitle_lines[i + 1])
            else:
                print(f"failed in {line} for {subtitle_file}")

    pil_images.append(get_full_image_with_text(current_image, texts))
    generate_pdf_from_images(pdf_name, pil_images)

    f.close()


def all_in_one(vf, method, bt=100, wpc=5000, nol=5):
    vtt_file_path, output_pdf_name = get_vtt_and_pdf_file_path(vf, method)
    if method == 1:
        convert_to_pdf_1(vf, vtt_file_path, output_pdf_name, bin_threshold=bt, white_pixel_count=wpc)
    elif method == 2:
        convert_to_pdf_2(vf, vtt_file_path, output_pdf_name, nol)
    elif method == 0:
        vtt_file_path, output_pdf_name = get_vtt_and_pdf_file_path(vf, method=1)
        convert_to_pdf_1(vf, vtt_file_path, output_pdf_name, bin_threshold=bt, white_pixel_count=wpc)
        vtt_file_path, output_pdf_name = get_vtt_and_pdf_file_path(vf, method=2)
        convert_to_pdf_2(vf, vtt_file_path, output_pdf_name, nol)


# if __name__ == "__main__":
#
#     d = defaultdict(list)
#     for k, v in ((k.lstrip('-'), v) for k, v in (a.split('=') for a in sys.argv[1:])):
#         d[k].append(v)
#
#     video_file = d.get("video_file", d["vf"])[0]
#     which = int(d.get("which", d.get("method", d.get("m", [1])))[0])
#
#     all_in_one(video_file, which)
#     print(f"Finished file {video_file}")
