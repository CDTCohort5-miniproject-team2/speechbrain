import audio_pipeline

wall_mics_array, server_closetalk, customer_closetalk = \
    audio_pipeline.get_test_sample(mics=("wall_mics", "server_closetalk", "customer_closetalk"))

customer_asr_model = audio_pipeline.initialise_asr()
server_asr_model = audio_pipeline.initialise_asr()

server_transcript = audio_pipeline.do_asr(server_closetalk, server_asr_model)
customer_transcript = audio_pipeline.do_asr(customer_closetalk, customer_asr_model)

print(server_transcript)
print(customer_transcript)