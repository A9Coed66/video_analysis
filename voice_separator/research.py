import numpy
import soundfile as sf
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input can be a URL or a local path
input_file = 'E:\\Project\\video_analysis\\3_一緒に食べよう！.wav'

# Resample to 8000Hz if needed
audio, sr = librosa.load(input_file, sr=None)
if sr != 8000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)
    resampled_file = 'resampled_8k.wav'
    sf.write(resampled_file, audio, 8000)
    input_file = resampled_file

separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer2_separation_temporal_8k')
result = separation(input_file)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)


#Chưa chạy được
# instantiate the pipeline
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained(
#   "pyannote/speaker-diarization-3.1",
#   use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# # run the pipeline on an audio file
# diarization = pipeline("audio.wav")

# # dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
