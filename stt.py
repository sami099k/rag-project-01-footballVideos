import whisper
import json
model = whisper.load_model('large-v2')
chunks = []
result = model.transcribe(
    audio = "/content/THE GALACTICOS vs LA MASIA [qpLR7ClWDI4].mp3",
    language = 'hi',task = 'translate',
    word_timestamps=False)

for segment in result['segments']:
    chunks.append({'start':segment['start'],'end':segment['end'],"text":segment['text']})


print(chunks)


with open('file_name.json','w') as f:
    json.dump(chunks,f)