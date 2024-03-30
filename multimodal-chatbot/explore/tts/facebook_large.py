from transformers import AutoProcessor, SeamlessM4TModel
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-large")
# let's load an audio sample from an Arabic speech corpus
from datasets import load_dataset
dataset = load_dataset("arabic_speech_corpus", split="test", streaming=True)
audio_sample = next(iter(dataset))["audio"]

# now, process it
# audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

text = "Once upon a time in a quaint village nestled amidst rolling hills and lush greenery, there existed a small brewery known as The Frothy Fox. This brewery was not just any ordinary establishment; it was renowned far and wide for its exceptional craft beers, \
    each brewed with passion, tradition, and a touch of magic. At the heart of The Frothy Fox stood its master brewer, a jovial and bearded man named Magnus. Magnus was not only a skilled artisan but \
    also a storyteller extraordinaire. He believed that every beer had a tale to tell, and it was his duty to unlock its secrets.\ One crisp autumn morning, as golden leaves danced in the gentle breeze,\
    Magnus decided it was time to create a beer unlike any other. Inspired by the colors of fall and the whispers of ancient forests, he embarked on a journey to craft the perfect brew."

# now, process some English test as well
text_inputs = processor(text = text, src_lang="eng", return_tensors="pt")
audio_array_from_text = model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
import soundfile as sf
sf.write("speech.wav", audio_array_from_text, samplerate=16000)