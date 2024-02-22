import torch
import sys
import numpy as np
from speechbrain.pretrained import SepformerSeparation as separator
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import stable_whisper
# NOTE TO TEAM: you will need to pip install optimum to use stable whisper

import prepare_kroto_data

sys.path.append("../DTLN-aec-main")  # so that we can import run_aec below
print("Importing run_aec")
import run_aec
print("run_aec imported.")

CATALOGUE = [
    # adding new recording sessions as they are available
    "test_kroto_data/18_12_2023",
    "test_kroto_data/01_02_24",
]
kroto_data = prepare_kroto_data.KrotoData(CATALOGUE[1])

def get_test_sample(audio_fstem="20240201_114729_scenario_28", timeslice=(2.0, 13.6), mics=("wall_mics", "server_closetalk")):
    """
    Get some sample audio arrays for testing
    :param audio_fstem:
    :param timeslice:
    :param mics: tuple of strings - according to channel mapping, e.g. ("wall_mics", "server_closetalk")
    :return: 2D arrays in the specific order
    """
    mic_arrays = []
    for mic_name in mics:
        mic_array = kroto_data.get_demo_audio_array(audio_fname=audio_fstem+".wav", downsampled=True,
                                                         timeslice=timeslice, channel_name=mic_name)
        mic_arrays.append(mic_array)
    return mic_arrays

class AudioPipeline:
    def __init__(self,
                 components=("aec", "asr"),
                 aec_size=512,
                 asr_model_name="whisper-medium.en",
                 long_transcription=True,
                 batch_input=False,
                 normalise_after_enhancing=True):
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
        for component in self.components:
            mapping[component][0]()
            self.speech_pipeline.append(mapping[component])

    def run_inference(self, target_1d_array, prompts, echo_cancel_1d_array=(), transcript_fname="demo"):
        print(prompts)
        if self.aec_model and (len(echo_cancel_1d_array) > 0):
            target_1d_array = self._do_aec(target_1d_array, echo_cancel_1d_array)
        if self.separator_model:
            target_1d_array = self._do_separating(target_1d_array)
        if self.enhancer_model:
            target_1d_array = self._do_enhancing(target_1d_array)

        transcript_object = self._do_asr(target_1d_array, prompts)

        timestamped_transcript_str = stable_whisper.result_to_tsv(transcript_object,
                                                                  filepath=None,
                                                                  segment_level=True,
                                                                  word_level=False)

        with open(f"{transcript_fname}.txt", "w") as f_obj:
            f_obj.write(timestamped_transcript_str)

        return timestamped_transcript_str

    def _initialise_aec(self):
        print("Initialising AEC model.")
        if self.aec_size not in [128, 256, 512]:
            raise ValueError("AEC component: model_size must be 128, 256, or 512.")
        aec_pretrained_fpath = f"../DTLN-aec-main/pretrained_models/dtln_aec_{self.aec_size}"
        interpreter1, interpreter2 = run_aec.initialise_interpreters(model=aec_pretrained_fpath)
        print("AEC model initialised.")
        self.aec_model = (interpreter1, interpreter2)

    def _initialise_separator(self):
        # TODO: code to set up source separation model goes here
        pass

    def _initialise_enhancer(self):
        print("Initialising enhancer model.")
        # model belongs to <class 'speechbrain.pretrained.interfaces.SepformerSeparation'>
        # TODO: query if we should be using this model for "enhancement" rather than "separation"
        #  maybe an alternative: https://github.com/facebookresearch/denoiser
        self.enhancer_model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                                     savedir="audio_pipeline_pretrained_models/sepformer-dns4-16k-enhancement")
        print("Enhancer model initialised.")

    def _initialise_asr(self, simple_stable_ts=True):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_size = self.asr_model_name.split("-")[1]

        if simple_stable_ts:
            self.asr_model = stable_whisper.load_hf_whisper(model_size)

        else:
            if "distil" in self.asr_model_name:
                model_id = "distil-whisper/" + self.asr_model_name
            else:
                model_id = "openai/" + self.asr_model_name
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)

            configurations = {"model": model, "tokenizer": processor.tokenizer, "feature_extractor": processor.feature_extractor,
                              "max_new_tokens": 128, "torch_dtype": torch_dtype, "device": device}
            if self.long_transcription:
                configurations["chunk_length_s"] = 15
            if self.batch_input:
                configurations["batch_size"] = 16

            self.asr_model = stable_whisper.load_hf_whisper(model_size,
                                                            device=device,
                                                            flash=False,
                                                            pipeline=pipeline("automatic-speech-recognition", **configurations))

    def _do_aec(self, target_array_1d, echo_array_1d):
        if len(target_array_1d.shape) > 1:
            target_array_1d = target_array_1d.squeeze(0)
        if len(echo_array_1d.shape) > 1:
            echo_array_1d = echo_array_1d.squeeze(0)

        return run_aec.process_audio(*self.aec_model, target_array_1d, echo_array_1d)

    def _do_enhancing(self, audio_array_1d):
        audio_tensor = torch.FloatTensor(audio_array_1d).unsqueeze(0)
        # audio_tensor has size (1, n_samples)
        est_sources = self.enhancer_model.separate_batch(audio_tensor)
        # est_sources has size (1, n_samples, 1)
        enhanced_array = est_sources[:, :, 0].detach().cpu().numpy().squeeze(0)
        # enhanced_array has shape (n_samples,)

        # check for clipping - adapted from DTLN-aec code
        if self.normalise_after_enhancing and (np.max(enhanced_array) > 1):
            enhanced_array = enhanced_array / np.max(enhanced_array) * 0.99
        return enhanced_array

    def _do_separating(self, audio_array):
        # TODO: implement
        return audio_array

    def _do_asr(self, audio_array_1d_or_fpath, prompts):
        # TODO: WORKING WITH 1D ARRAY FOR NOW - CAN BE MODIFIED LATER TO TAKE BATCH INPUTS
        if isinstance(audio_array_1d_or_fpath, np.ndarray) and len(audio_array_1d_or_fpath.shape) == 2:
            audio_array_1d_or_fpath = audio_array_1d_or_fpath.squeeze(0)

        if "distil" in self.asr_model_name:
            return self.asr_model(audio_array_1d_or_fpath, return_timestamps=True)
        else:
            if prompts:
                print("Implementing Prompt engineering")
                # https://github.com/jianfch/stable-ts/tree/main
                return self.asr_model.transcribe(audio_array_1d_or_fpath, initial_prompt = prompts)
            else: 
                # https://github.com/jianfch/stable-ts/tree/main
                return self.asr_model.transcribe(audio_array_1d_or_fpath)

def main():
    wall_mics, customer_closetalk, server_closetalk = \
        get_test_sample("20240201_104809_scenario_10",
                        timeslice=(0, 0),
                        mics=("wall_mics", "customer_closetalk", "server_closetalk"))

    wall_mic, customer_closetalk, server_closetalk = wall_mics[0, :], \
                                                     customer_closetalk.squeeze(0), \
                                                     server_closetalk.squeeze(0)

    server_side_pipeline = AudioPipeline(components=("aec", "asr"))

    server_transcript = server_side_pipeline.run_inference(target_1d_array=server_closetalk,
                                                           echo_cancel_1d_array=customer_closetalk,
                                                           transcript_fname="server_side_demo", 
                                                           prompts = "ShefBurger")

    customer_side_pipeline = AudioPipeline(components=("aec", "asr"))

    customer_transcript = customer_side_pipeline.run_inference(target_1d_array=wall_mic,
                                                               echo_cancel_1d_array=server_closetalk,
                                                               transcript_fname="customer_side_demo",
                                                               prompts = "Mambo Combo")

    print(server_transcript)
    print(customer_transcript)


if __name__ == "__main__":
    main()
