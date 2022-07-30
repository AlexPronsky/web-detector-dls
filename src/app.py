import base64
from flask import Flask, render_template, request
from PIL import Image
from service.ssd_detector import SSDDetector
from service.utils import draw_detections, prepare_tensor

debug_mode = True
app = Flask(__name__)
app.debug = debug_mode

ssd_detector = SSDDetector()
image_path = 'tmp/image.png'


@app.route('/', methods=('GET', 'POST'))
def process():
    if 'image' in request.files:
        file = request.files['image']
        img = Image.open(file.stream)
        img.save(image_path)

        inputs = ssd_detector.prepare_inputs([image_path])
        tensors = prepare_tensor(inputs)
        detections = ssd_detector.detect(tensors)

        results_per_input = ssd_detector.decode_results(detections)
        best_results_per_input = ssd_detector.pick_best(results_per_input)

        image_bytes = draw_detections(inputs[0], best_results_per_input[0], ssd_detector.classes_to_labels)
        image_base64 = base64.b64encode(image_bytes).decode()

        return render_template('index.html', img_data=image_base64)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=debug_mode)
