import io
import wave
import whisper
import asyncio
import websockets
import numpy as np
from pydub import AudioSegment

model = whisper.load_model('base')

def convert_bytes_to_whisper_format(audio_segment):
    if audio_segment.frame_rate != 16000: # 16kHz
        audio_segment = audio_segment.set_frame_rate(16000)
    if audio_segment.sample_width != 2:   # int16
        audio_segment = audio_segment.set_sample_width(2)
    if audio_segment.channels != 1:       # mono
        audio_segment = audio_segment.set_channels(1)        
    audio = np.array(audio_segment.get_array_of_samples())
    audio = audio.astype(np.float32) / 32768.0
    return audio

def extract_wave_header_and_params(wave_bytes):
    with io.BytesIO(wave_bytes) as wave_bytes_io:
        with wave.open(wave_bytes_io, 'rb') as wave_file:
            header = wave_file.getparams()
    return header

def add_header_to_chunk(audio_bytes, header):
    # Creating a BytesIO object to write the wave header and chunk wave bytes
    with io.BytesIO() as wave_bytes_io:
        # Using wave module to write header and data into the BytesIO object
        with wave.open(wave_bytes_io, 'wb') as wave_file:
            wave_file.setparams(header)
            wave_file.writeframes(audio_bytes)
        
    # Getting the complete wave bytes (header + data)
    complete_wave_bytes = wave_bytes_io.getvalue()
    return complete_wave_bytes

async def audio_receiver(websocket, path, global_header=r''):
    async for message in websocket:
        with open(f'./backend/data/audio_{message[0]}.wav', 'wb') as f:
            f.write(message)
        try:
            audio_bytes = io.BytesIO(message)
            audio_segment = AudioSegment.from_file(audio_bytes, format='wav')
            audio_to_transcript = convert_bytes_to_whisper_format(audio_segment)

            transcript = model.transcribe(audio_to_transcript)
            print(transcript['text'])
        except:
            print('No header')

    # Send back the transcription or acknowledge receipt
    await websocket.send("Transcription received or Transcript text")

start_server = websockets.serve(audio_receiver, "localhost", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
