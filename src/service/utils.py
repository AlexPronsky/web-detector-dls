import io
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches


matplotlib.use('Agg')


def prepare_tensor(inputs, fp16=False):
    NHWC = np.array(inputs)
    NCHW = np.swapaxes(np.swapaxes(NHWC, 1, 3), 2, 3)
    tensor = torch.from_numpy(NCHW)
    tensor = tensor.contiguous()
    # tensor = tensor.cuda()
    tensor = tensor.float()
    if fp16:
        tensor = tensor.half()
    return tensor


def draw_detections(input_image, best_results, classes_to_labels):
    fig, ax = plt.subplots(1)

    # Show original, denormalized image...
    image = input_image / 2 + 0.5
    ax.imshow(image)

    # Draw detection boxes
    bboxes, classes, confidences = best_results
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx] * 100),
                bbox=dict(facecolor='white', alpha=0.5))
        # ax.axis('tight')
        ax.axis('off')

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    # img = Image.open(buf)
    return buf.getvalue()


# def visualize_detections(inputs, best_results_per_input, classes_to_labels):
#     for image_idx in range(len(best_results_per_input)):
#         fig, ax = plt.subplots(1)
#
#         # Show original, denormalized image...
#         image = inputs[image_idx] / 2 + 0.5
#         ax.imshow(image)
#
#         # ...with detections
#         bboxes, classes, confidences = best_results_per_input[image_idx]
#         for idx in range(len(bboxes)):
#             left, bot, right, top = bboxes[idx]
#             x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
#             rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)
#             ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx] * 100),
#                     bbox=dict(facecolor='white', alpha=0.5))
#     plt.show()
