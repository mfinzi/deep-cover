import torch
import numpy as np
import streamlit as st
from encode import encode_long_text, decode_long_text

@st.cache(ignore_hash=True)
def load_models():
    en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    fr2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2fr.cuda();
    fr2en.cuda();
    return (en2fr,fr2en)
models = load_models()
phrase="Japanese researchers began studying transistors three months after they were invented at America’s Bell Labs in 1947. Japanese companies then used transistors and other electronic parts and components to produce radios, television sets, Sony Walkmans, video cassette recorders, and computers. As the yen appreciated by 60% following the 1985 Plaza Accord, Japanese companies lost competitiveness in final electronics goods and moved upstream in electronics value chains. They focused on exporting electronic parts and components and capital goods to producers of final electronics goods abroad. "
cover_text = st.text_area("Cover Text", phrase)

secret_message = st.text_area("Secret Message","secret code here: blah blah")
T = st.slider("Temperature", min_value=0.7, max_value=1.5, value=1.05, step=None, format=None)
topk = st.slider("topk", min_value=5, max_value=200, value=25, step=5, format=None)

encoded_text, bits_encoded = encode_long_text(phrase,secret_message,models,temperature=T,sampling_topk=topk)
st.write('Cover + Payload:')
st.write(encoded_text)
st.write(f'{bits_encoded} payload bits delivered')

if st.checkbox('Decode?'):
    decoded_text = decode_long_text(phrase,encoded_text,models,temperature=T,sampling_topk=topk)
    st.write(decoded_text)