import torch
import speechbrain as sb
import sys
import numpy as np
from speechbrain.pretrained import SepformerSeparation as separator
import stable_whisper
# NOTE TO TEAM: you will need to pip install optimum to use stable whisper
# !pip install ssspy
from ssspy.bss.iva import AuxLaplaceIVA
import source_sep
import load_kroto_data
from time import time


sys.path.append("../DTLN-aec-main")  # so that we can import run_aec below
print("Importing run_aec")
import run_aec
print("run_aec imported.")

def get_test_sample(kroto_data_obj, audio_fstem="20240201_114729_scenario_28", timeslice=(2.0, 13.6), mics=("wall_mics", "server_closetalk")):
    """
    Get some sample audio arrays for testing
    :param audio_fstem:
    :param kroto_data_obj: an instance of load_kroto_data.RawKrotoData
    :param timeslice: in seconds
    :param mics: tuple of strings - according to channel mapping, e.g. ("wall_mics", "server_closetalk")
    :return: 2D arrays in the specific order
    """
    mic_arrays = []
    for mic_name in mics:
        mic_array = kroto_data_obj.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                         timeslice=timeslice, channel_name=mic_name)
        mic_arrays.append(mic_array)
    return mic_arrays

class AudioPipeline:
    def __init__(self,
                 components=("aec", "asr"),
                 aec_size=512,
                 asr_model_name="whisper-large-v3",
                 long_transcription=True,
                 batch_input=False,
                 normalise_after_enhancing=True):
        """

        :param components:
        :param aec_size:
        :param asr_model_name: ASR model to use e.g. "whisper-large-v3", "whisper-tiny.en"
        :param long_transcription:
        :param batch_input:
        :param normalise_after_enhancing:
        """
        self.components = components
        self.aec_size = aec_size
        self.long_transcription = long_transcription
        self.batch_input = batch_input
        self.asr_model_name = asr_model_name
        self.normalise_after_enhancing = normalise_after_enhancing

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

        :param target_array:
        :param echo_cancel_array:
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

            elif component_name == "separator":
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
        print("Initialising enhancer model.")
        # model belongs to <class 'speechbrain.pretrained.interfaces.SepformerSeparation'>
        self.enhancer_model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                                     savedir="audio_pipeline_pretrained_models/sepformer-dns4-16k-enhancement")
        print("Enhancer model initialised.")

    def _initialise_asr(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_size = self.asr_model_name.split("-")
        self.asr_model = stable_whisper.load_hf_whisper(model_size[1])

    def _do_aec(self, target_array_nd, echo_array_nd, batch=False):
        # NOTE TO TEAM2: our AEC doesn't support batch-processing, we will need to manually configure this,
        if batch:
            results = np.zeros(target_array_nd.shape)
            for i, (target_array, echo_array) in enumerate(zip(target_array_nd, echo_array_nd)):
                results[i] = run_aec.process_audio(*self.aec_model, target_array, echo_array)
            return results  # 2D array

        else:
            if len(target_array_nd.shape) > 1:
                target_array_nd = target_array_nd.squeeze(0)
            if len(echo_array_nd.shape) > 1:
                echo_array_nd = echo_array_nd.squeeze(0)

            return run_aec.process_audio(*self.aec_model, target_array_nd, echo_array_nd)  # 1D array

    def _do_enhancing(self, audio_array, batch=False):
        # currently takes either a 1D array (n_samples,), or 2D (n_batch, n_samples,) if batch=True
        audio_tensor = torch.FloatTensor(audio_array)
        if (not batch) and len(audio_array.shape > 1):
            audio_tensor = torch.unsqueeze(audio_tensor, 0)
        # audio_tensor has size (n_batch, n_samples)
        est_sources = self.enhancer_model.separate_batch(audio_tensor)
        # est_sources has size (n_batch, n_samples, 1)
        enhanced_arrays = est_sources[:, :, 0].detach().cpu().numpy()
        # enhanced_arrays has shape (n_batch, n_samples,)

        for array_idx, _ in enumerate(enhanced_arrays):
            # check for clipping - adapted from DTLN-aec code
            if self.normalise_after_enhancing and (np.max(enhanced_arrays[array_idx]) > 1):
                enhanced_arrays[array_idx] = enhanced_arrays[array_idx] / np.max(enhanced_arrays[array_idx]) * 0.99
        if not batch:
            enhanced_arrays = enhanced_arrays.squeeze(0)
        return enhanced_arrays

    def _do_separating(self, audio_array_1d):
        stereo_audio = source_sep.make_stereo(audio_array_1d)
        return source_sep.do_source_sep(self.separator_model, stereo_audio)[1]  # return the second source

    def _do_asr(self, audio_array_1d_or_fpath):

        if isinstance(audio_array_1d_or_fpath, np.ndarray) and len(audio_array_1d_or_fpath.shape) == 2:
            audio_array_1d_or_fpath = audio_array_1d_or_fpath.squeeze(0)
        # https://github.com/jianfch/stable-ts/tree/main
        return self.asr_model.transcribe(audio_array_1d_or_fpath)

def main():
    wall_mics, customer_closetalk, server_closetalk = \
        get_test_sample(load_kroto_data.RawKrotoData("kroto_data"),
                        "20240201_104809_scenario_10",
                        timeslice=(0, 0),
                        mics=("wall_mics", "customer_closetalk", "server_closetalk"))

    wall_mic, customer_closetalk, server_closetalk = wall_mics[0, :], \
                                                     customer_closetalk.squeeze(0), \
                                                     server_closetalk.squeeze(0)

    server_side_pipeline = AudioPipeline(components=("aec", "asr"))

    server_output_audio, server_transcript, _ = server_side_pipeline.run_inference(target_array=server_closetalk,
                                                                                   echo_cancel_array=customer_closetalk)

    customer_side_pipeline = AudioPipeline(components=("aec", "enhancer", "separator", "asr"))

    customer_output_audio, customer_transcript, _ = customer_side_pipeline.run_inference(target_array=wall_mic,
                                                                                         echo_cancel_array=server_closetalk)

    print(server_transcript)
    print(customer_transcript)

if __name__ == "__main__":
    main()
