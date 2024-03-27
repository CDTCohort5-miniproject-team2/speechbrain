import torch
import speechbrain as sb
import sys
import numpy as np
from speechbrain.pretrained import SepformerSeparation as separator
import stable_whisper
from ssspy.bss.iva import AuxLaplaceIVA
import source_sep
import load_kroto_data
from time import time

# !pip install optimum
# !pip install ssspy

sys.path.append("../DTLN-aec-main")  # so that we can import run_aec below
print("Importing run_aec")
import run_aec
print("run_aec imported.")

sys.path.append("../DTLN-master")
print("Importing enhancer module")
from DTLN_model import DTLN_model
import run_evaluation as enhancer_module
print("Enhancer module imported.")

class AudioPipeline:
    def __init__(self,
                 components=("aec", "asr"),
                 aec_size=512,
                 asr_model_name="whisper-medium.en",
                 long_transcription=True,):
        """
        A class that integrates all speech-side components into a customisable pipeline
        that runs inference on input audio arrays
        :param components: (tuple of str) a list of components (order-sensitive) to be included in the pipeline
        - aec / enhancer / separator / asr
        :param aec_size: (int) 128, 256 or 512 - model size of our AEC component, see report for further details
        :param asr_model_name: ASR model to use e.g. "whisper-large-v3", "whisper-tiny.en"
        :param long_transcription:
        """
        self.components = components
        self.aec_size = aec_size
        self.long_transcription = long_transcription
        self.asr_model_name = asr_model_name

        self.aec_model, self.separator_model, self.enhancer_model, self.asr_model = None, None, None, None

        mapping = {"aec": (self._initialise_aec, self._do_aec),
                   "separator": (self._initialise_separator, self._do_separating),
                   "enhancer": (self._initialise_enhancer, self._do_enhancing),
                   "asr": (self._initialise_asr, self._do_asr),
                   }

        self.speech_pipeline = []
        for component_name in self.components:
            (mapping[component_name])[0]()
            self.speech_pipeline.append((component_name, mapping[component_name][1]))

    def run_inference(self, target_array, echo_cancel_array=None):
        """
        Runs input audio through each component in the pipeline and returns the processed array and predicted transcript
        :param target_array: 1D array - target signal to be processed
        :param echo_cancel_array: 1D array - echo signal to be cancelled from the target
        :return: processed array, transcription (str), and a list of durations (s) that each component in the pipeline took
        """
        timestamped_transcript_str = None
        master_start_time = time()
        component_wise_times = []

        for component_name, component_function in self.speech_pipeline:
            component_wise_start_time = time()
            if component_name == "aec":
                target_array = self._do_aec(target_array, echo_cancel_array)

            elif component_name == "separator":
                target_array = self._do_separating(target_array)

            elif component_name == "enhancer":
                target_array = self._do_enhancing(target_array)

            elif component_name == "asr":
                transcript_object = self._do_asr(target_array)

                timestamped_transcript_str = stable_whisper.result_to_tsv(transcript_object,
                                                                          filepath=None,
                                                                          segment_level=True,
                                                         word_level=False)
            component_wise_times.append(time()-component_wise_start_time)
        component_wise_times.insert(0, time()-master_start_time)

        return target_array, timestamped_transcript_str, component_wise_times

    def _initialise_aec(self):
        print("Initialising AEC model.")
        if self.aec_size not in [128, 256, 512]:
            raise ValueError("AEC component: model_size must be 128, 256, or 512.")
        aec_pretrained_fpath = f"../DTLN-aec-main/pretrained_models/dtln_aec_{self.aec_size}"
        interpreter1, interpreter2 = run_aec.initialise_interpreters(model=aec_pretrained_fpath)
        print("AEC model initialised.")
        self.aec_model = (interpreter1, interpreter2)

    def _initialise_separator(self):
        self.separator_model = AuxLaplaceIVA()

    def _initialise_enhancer(self):
        # using the DTLN model
        weights_fpath = "../DTLN-master/pretrained_model/model.h5"

        self.enhancer_model = DTLN_model()
        self.enhancer_model.build_DTLN_model()
        self.enhancer_model.model.load_weights(weights_fpath)

    def _initialise_asr(self):
        model_size = self.asr_model_name.split("-")
        self.asr_model = stable_whisper.load_hf_whisper(model_size[1])

    def _do_aec(self, target_array_nd, echo_array_nd):
        if len(target_array_nd.shape) > 1:
            target_array_nd = target_array_nd.squeeze(0)
        if len(echo_array_nd.shape) > 1:
            echo_array_nd = echo_array_nd.squeeze(0)

            return run_aec.process_audio(*self.aec_model, target_array_nd, echo_array_nd)  # 1D array

    def _do_enhancing(self, audio_array):
        return enhancer_module.process_audio_array(self.enhancer_model.model, audio_array)

    def _do_separating(self, audio_array_1d):
        stereo_audio = source_sep.make_stereo(audio_array_1d)
        return source_sep.do_source_sep(self.separator_model, stereo_audio)[1]  # return the second source

    def _do_asr(self, audio_array_1d_or_fpath):
        if isinstance(audio_array_1d_or_fpath, np.ndarray) and len(audio_array_1d_or_fpath.shape) == 2:
            audio_array_1d_or_fpath = audio_array_1d_or_fpath.squeeze(0)
        # https://github.com/jianfch/stable-ts/tree/main
        return self.asr_model.transcribe(audio_array_1d_or_fpath)

def main():
    server_side_pipeline = AudioPipeline(components=("aec", "asr"))
    customer_side_pipeline = AudioPipeline(components=("aec", "enhancer", "separator", "asr"))

    rawdata = load_kroto_data.RawKrotoData("kroto_data")
    torch_dataset = rawdata.get_torch_dataset()
    for i, (scenario_id, server_closetalk, customer_closetalk, top_centre_wall_mic_array) in enumerate(torch_dataset):

        server_output_audio, server_transcript, _ = server_side_pipeline.run_inference(target_array=server_closetalk,
                                                                                       echo_cancel_array=customer_closetalk)

        customer_output_audio, customer_transcript, _ = customer_side_pipeline.run_inference(target_array=top_centre_wall_mic_array,
                                                                                             echo_cancel_array=server_closetalk)

        print(server_transcript)
        print(customer_transcript)

        if i == 2:
            break

if __name__ == "__main__":
    main()
