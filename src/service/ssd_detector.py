import torch


class SSDDetector:

    def __init__(self):
        self.precision = 'fp32'
        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd',
                                        model_math=self.precision, pretrained=False)
        self.ssd_model.load_state_dict(torch.load('../models/ssd_model.pt'))
        self.ssd_model.eval()

        # self.ssd_model = torch.load('../models/ssd_model.pt')
        self.ssd_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.classes_to_labels = self.ssd_utils.get_coco_object_dictionary()

    def detect(self, input_tensors):
        with torch.no_grad():
            return self.ssd_model(input_tensors)

    def prepare_inputs(self, uris):
        inputs = [self.ssd_utils.prepare_input(uri) for uri in uris]
        return inputs

    def prepare_tensors(self, inputs):
        tensors = self.ssd_utils.prepare_tensor(inputs, self.precision == 'fp16')
        return tensors

    def decode_results(self, detections):
        return self.ssd_utils.decode_results(detections)

    def pick_best(self, results_per_input):
        return [self.ssd_utils.pick_best(results, 0.30) for results in results_per_input]
