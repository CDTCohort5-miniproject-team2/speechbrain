from audio_pipeline import *

def aec_test(play_out=True):
    wall_mics_array, server_closetalk_array = get_test_sample()
    single_wall_mic_array = wall_mics_array[1, :]
    print(single_wall_mic_array.shape)
    print(server_closetalk_array.shape)
    interpreter1, interpreter2 = initialise_aec()
    print("Performing AEC.")
    post_aec_array = do_aec(interpreter1, interpreter2, single_wall_mic_array, server_closetalk_array.squeeze(0))
    print("AEC done.")
    if play_out:
        print("Playing pre-AEC wall mic")
        prepare_kroto_data.play_audio_array(single_wall_mic_array)
        print("Playing post-AEC wall mic")
        prepare_kroto_data.play_audio_array(post_aec_array)

def beamforming_and_aec_test(play_out=True, aec_first=False):
    wall_mics_array, server_closetalk_array = get_test_sample()
    interpreter1, interpreter2 = initialise_aec()
    initialised_beamformer = initialise_beamforming(channels=(4, 5, 6, 7, 8), mode="doa")

    if aec_first:
        processed_array = first_aec_then_beamforming(interpreter1, interpreter2,
                                                     wall_mics_array, server_closetalk_array,
                                                     initialised_beamformer)
    else:
        # do beamforming first
        processed_array = first_beamforming_then_aec(interpreter1, interpreter2,
                                                     wall_mics_array, server_closetalk_array,
                                                     initialised_beamformer)

    if play_out:
        print("Playing pre-AEC/beamforming wall mic")
        prepare_kroto_data.play_audio_array(wall_mics_array)
        print("Playing post-AEC/beamforming wall mic")
        prepare_kroto_data.play_audio_array(processed_array)

def enhancing_test(play_out=True):
    """
    WARNING: THIS ENHANCER MODEL SEEMS TO OVER-AMPLIFY THE SIGNAL.
    WE MAY NEED TO DO SOME POST-PROCESSING E.G. TO AVOID CLIPPING.
    :param play_out: if true, plays out the pre- and post-enhanced audio for comparison.
    Make sure your speaker/headphone volume is turned down sufficiently before testing
    :return:
    """
    model = initialise_enhancing()
    wall_mics_array, server_closetalk_array = get_test_sample()
    single_wall_mic_array = wall_mics_array[1, :]
    print("Enhancing array.")
    enhanced_array_normalised = do_enhancing(single_wall_mic_array, model, normalise=True)
    print("Array enhanced.")

    if play_out:
        print("Playing pre-enhanced array")
        prepare_kroto_data.play_audio_array(single_wall_mic_array)
        print("Playing enhanced array")
        prepare_kroto_data.play_audio_array(enhanced_array_normalised)

def asr_test(play_out=False):
    asr_model = initialise_asr()
    wall_mics_array, server_closetalk_array = get_test_sample()
    single_wall_mic_array = wall_mics_array[1, :]
    if play_out:
        print("Playing pre-enhanced single-channel wall mic")
        prepare_kroto_data.play_audio_array(single_wall_mic_array)

    asr_result = do_asr(single_wall_mic_array, asr_model)
    print(asr_result)


if __name__ == "__main__":
    play_out = False
    print("Running tests")
    aec_test(play_out=play_out)
    beamforming_and_aec_test(play_out=play_out, aec_first=False)
    beamforming_and_aec_test(play_out=play_out, aec_first=True)
    enhancing_test(play_out=False)
    asr_test(play_out=False)
    print("Tests completed.")


