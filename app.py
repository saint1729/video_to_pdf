import os
import time

from flask import Flask, render_template, request, send_file

import video_to_pdf

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        video_file = request.files['video_file']
        vtt_file = request.files['vtt_file']

        timestamp = int(time.time() * 1000)
        os.mkdir(app.config['UPLOAD_FOLDER'] + str(timestamp) + "/")

        video_file.save(app.config['UPLOAD_FOLDER'] + str(timestamp) + "/" + "temp.mp4")
        vtt_file.save(app.config['UPLOAD_FOLDER'] + str(timestamp) + "/" + "temp.vtt")

        vf = app.config['UPLOAD_FOLDER'] + str(timestamp) + "/" + "temp.mp4"
        method = 1

        video_to_pdf.all_in_one(vf, method)
        vtt_file_name, server_pdf_name = video_to_pdf.get_vtt_and_pdf_file_path(vf, 1)
        output_pdf_name = os.path.splitext(video_file.filename)[0] + ".pdf"

        return send_file(server_pdf_name, download_name=output_pdf_name, as_attachment=True)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
