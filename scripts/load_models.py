import torch
# precision = 'fp32'
# ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd',
#                            model_math=precision, pretrained=False)
#
# checkpoint = torch.hub.load_state_dict_from_url(
#     'https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt',
#     map_location="cpu"
# )
# ssd_model.load_state_dict(checkpoint['model'])
# torch.save(ssd_model, '../models/ssd_model.pt')
#
# utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

checkpoint = torch.hub.load_state_dict_from_url(
    'https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt',
    map_location="cpu"
)

torch.save(checkpoint['model'], '../models/ssd_model.pt')
