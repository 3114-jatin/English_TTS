import torch
from transformers import SpeechT5ForTextToSpeech

class CustomSpeechT5(SpeechT5ForTextToSpeech):
    def __init__(self, config):
        super().__init__(config)

        # Modify Conv1d layers within the speech decoder postnet
        for name, layer in self.speech_decoder_postnet.named_children():
            if isinstance(layer, torch.nn.Conv1d):
                # Ensure the kernel size does not exceed input size
                input_size = layer.in_channels  # Adjust based on actual input size

                # Dynamically adjust the kernel size based on input size
                if input_size >= 5:
                    layer.kernel_size = (5,)
                elif input_size >= 3:
                    layer.kernel_size = (3,)
                else:
                    layer.kernel_size = (1,)
 # Fallback for very small input sizes

                # Set stride and padding to avoid dimension issues
                layer.stride = (1,)
                layer.padding = (layer.kernel_size[0] // 2,)  # Set padding to maintain output size

    def custom_method(self):
        pass

def load_custom_model():
    from transformers import SpeechT5Config
    config = SpeechT5Config.from_pretrained("microsoft/speecht5_tts")
    custom_model = CustomSpeechT5(config)
    return custom_model
