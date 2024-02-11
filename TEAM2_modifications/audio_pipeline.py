import torch
import sys
import numpy as np

from speechbrain.pretrained import SepformerSeparation as separator
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# from faster_whisper import WhisperModel

import prepare_kroto_data
import _TEAM2_beamforming as bf


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

def initialise_audio_pipeline():
    # TODO: extend function to allow for various optional arguments/customisations
    initialised_beamformer = initialise_beamforming(channels=(4, 5, 6, 7, 8), mode="doa")
    interpreter1, interpreter2 = initialise_aec(model_size=128)
    enhancer_model = initialise_enhancing()
    asr_model = initialise_asr(model_name="distil-large-v2", long_transcription=False, batch_input=False)

    return initialised_beamformer, interpreter1, interpreter2, enhancer_model, asr_model

def initialise_beamforming(channels=(4, 5, 6, 7, 8), mode="doa"):
    """
    This is refactored from _TEAM2_beamforming.py
    :param channels: tuple of int (4, 5, 6, 7, 8)
    :return:
    """
    stft, istft = bf.STFT(sample_rate=bf.SAMPLING_RATE, n_fft=1200), bf.ISTFT(sample_rate=bf.SAMPLING_RATE, n_fft=1200)
    covariance = bf.Covariance()
    delaysum = bf.DelaySum()
    if mode == "doa":
        srpphat_or_gccphat = bf.SrpPhat(mics=bf.get_mic_locations(channels=channels))
    elif mode == "tdoa":
        srpphat_or_gccphat = bf.GccPhat()
    else:
        raise NotImplementedError("Specify an implemented method of beamforming.")

    initialised_beamformer = (stft, istft, covariance, delaysum, srpphat_or_gccphat)
    return initialised_beamformer

def do_beamforming(audio_array_2d, mode, channels, initialised_beamformer):
    audio_tensor = torch.FloatTensor(audio_array_2d.T).unsqueeze(0)
    beamformed_tensor = bf.delaysum_beamforming(audio_tensor, mode=mode, channels=channels,
                                                pre_instantiated=initialised_beamformer)
    beamformed_array = beamformed_tensor.squeeze(dim=(0, -1)).numpy()
    return beamformed_array

def initialise_aec(model_size=512):
    """
    Set up the DTLN AEC
    :param model_size: int - must be 128, 256 or 512
    :return: interpreter1, interpreter2 - these will be used by the do_aec function for performing AEC
    """
    print("Initialising AEC model.")
    if model_size not in [128, 256, 512]:
        raise ValueError("AEC component: model_size must be 128, 256, or 512.")
    aec_pretrained_fpath = f"../DTLN-aec-main/pretrained_models/dtln_aec_{model_size}"
    interpreter1, interpreter2 = run_aec.initialise_interpreters(model=aec_pretrained_fpath)
    print("AEC model initialised.")
    return interpreter1, interpreter2

def do_aec(interpreter1, interpreter2, wall_mic_array_1d, server_closetalk_array_1d):
    """
    :param interpreter1: interpreter1 obtained from calling initialise_aec
    :param interpreter2: interpreter2 obtained from calling initialise_aec
    :param wall_mic_array_1d:
    :param server_closetalk_array_1d:
    :return: 1d-array after AEC
    """
    if len(server_closetalk_array_1d.shape) > 1:
        server_closetalk_array_1d = server_closetalk_array_1d.squeeze(0)

    return run_aec.process_audio(interpreter1, interpreter2, wall_mic_array_1d, server_closetalk_array_1d)

def initialise_enhancing():
    print("Initialising enhancer model.")
    # model belongs to <class 'speechbrain.pretrained.interfaces.SepformerSeparation'>
    model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement",
                                   savedir="audio_pipeline_pretrained_models/sepformer-dns4-16k-enhancement")
    print("Enhancer model initialised.")
    return model

def do_enhancing(audio_array_1d, model, normalise=True):
    # TODO: extend this function to work with audio as file in addition to as a numpy array
    audio_tensor = torch.FloatTensor(audio_array_1d).unsqueeze(0)
    # audio_tensor has size (1, n_samples)
    est_sources = model.separate_batch(audio_tensor)
    # est_sources has size (1, n_samples, 1)
    enhanced_array = est_sources[:, :, 0].detach().cpu().numpy().squeeze(0)
    # enhanced_array has shape (n_samples,)

    # check for clipping - adapted from DTLN-aec code
    if normalise and (np.max(enhanced_array) > 1):
        enhanced_array = enhanced_array / np.max(enhanced_array) * 0.99
    return enhanced_array

def initialise_asr(model_name="distil-large-v2", long_transcription=False, batch_input=False):
    # set long_transcription to True if the intended audio/file for inference is >30 seconds
    # set batch_input to True if intending to use the model on batches of audio or files
    if "distil" in model_name:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "distil-whisper/" + model_name
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        configurations = {"model": model, "tokenizer": processor.tokenizer, "feature_extractor": processor.feature_extractor,
                          "max_new_tokens": 128, "torch_dtype": torch_dtype, "device": device}
        if long_transcription:
            configurations["chunk_length_s"] = 15
        if batch_input:
            configurations["batch_size"] = 16

        return pipeline("automatic-speech-recognition", **configurations)

    else:
        torch_dtype = "fp16" if torch.cuda.is_available() else "int8"
        # fallback to faster-whisper, not yet successfully implemented

        # fyi: CPUs can't seem to work on float16 faster-whisper
        # see https://github.com/SYSTRAN/faster-whisper/issues/65
        print("Initialising ASR model.")
        asr_model = None
        # asr_model = WhisperModel(model_name, compute_type=torch_dtype)
        print("ASR model initialised.")
        return asr_model
def do_asr(audio_array_1d_or_fpath, asr_model, distil_whisper=True):
    # note for ref: faster-whisper WhisperModel's transcribe method can process audio either as file or as array
    # TODO: distil_whisper=False was intended for faster-whisper but needs fixing - this is currently resulting in
    #  "OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized."

    if isinstance(audio_array_1d_or_fpath, np.ndarray) and len(audio_array_1d_or_fpath.shape) == 2:
        audio_array_1d_or_fpath = audio_array_1d_or_fpath.squeeze(0)

    if distil_whisper:
        return asr_model(audio_array_1d_or_fpath, return_timestamps=True)
    else:
        return asr_model.transcribe(audio_array_1d_or_fpath)

def first_beamforming_then_aec(interpreter1, interpreter2, audio_array_nd, server_closetalk, initialised_beamformer):
    post_beamform_array = do_beamforming(audio_array_nd, "doa", channels=(4, 5, 6, 7, 8), initialised_beamformer=initialised_beamformer)
    return do_aec(interpreter1, interpreter2, post_beamform_array, server_closetalk)

def first_aec_then_beamforming(interpreter1, interpreter2, audio_array_nd, server_closetalk, initialised_beamformer):
    # breaking the wall_mics apart and then run aec on each of these
    post_aec_array = np.array(
        [do_aec(interpreter1, interpreter2, array_1d, server_closetalk)
         for array_1d in audio_array_nd]
    )
    # it would be good to parallelise this, but not sure if it could be done.
    # alternatively, we need a good multichannel acoustic echo cancellation library

    return do_beamforming(post_aec_array, "doa", channels=(4, 5, 6, 7, 8), initialised_beamformer=initialised_beamformer)







def main():
    pass

if __name__ == "__main__":
    main()

