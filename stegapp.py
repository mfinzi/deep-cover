import torch
import numpy as np
import streamlit as st
from encode import encode_long_text, decode_long_text,colorize
import difflib
@st.cache(ignore_hash=True)
def load_models():
    en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    fr2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2fr.cuda();
    fr2en.cuda();
    return (en2fr,fr2en)

def token2color(token,added=True):
    if token[0]=='+':
        return '```{}```'.format(token[2:]) if added else ''#" b`{}`b ".format(token[1:]) if (added) else ''#colorize(token[1:],'green') if added else ''
    elif token[0]=='-':
        return '' if added else '```{}```'.format(token[2:])#`{}".format(token[1:])#colorize(token[1:],'red')
    elif token[0]=='?':
        return ''
    else:
        return token[2:]

models = load_models()
T = st.sidebar.slider("Temperature", min_value=0.7, max_value=1.5, value=1.05, step=None, format=None)
topk = st.sidebar.slider("topk", min_value=5, max_value=200, value=25, step=5, format=None)

decode = st.sidebar.checkbox('Decode?')
if not decode:
    st.markdown('# Encode')
else:
    st.markdown('# Decode')
phrase="Japanese researchers began studying transistors three months after they were invented at America’s Bell Labs in 1947. Japanese companies then used transistors and other electronic parts and components to produce radios, television sets, Sony Walkmans, video cassette recorders, and computers. As the yen appreciated by 60% following the 1985 Plaza Accord, Japanese companies lost competitiveness in final electronics goods and moved upstream in electronics value chains. They focused on exporting electronic parts and components and capital goods to producers of final electronics goods abroad. "
cover_text = st.text_area("Cover Text", phrase)
if not decode:
    
    secret_message = st.text_area("Secret Message","secret code here: blah blah")
    encoded_text, bits_encoded, otherlang_text = encode_long_text(cover_text,secret_message,models,temperature=T,sampling_topk=topk)


    diff = st.sidebar.checkbox('Diff?',False)
    if diff:
        diff = list(difflib.Differ().compare(cover_text.split(' '),encoded_text.split(' ')))
        highlighted_diff_p = ' '.join([token2color(tok,True) for tok in diff])
        highlighted_diff_m = ' '.join([token2color(tok,False) for tok in diff]).replace('```','~~')
        st.markdown('## Cover:')
        st.markdown(highlighted_diff_m)
        st.markdown('## Cover + Payload:')
        st.markdown(highlighted_diff_p)
        
    else:
        st.markdown('## Cover + Payload:')
        st.write(encoded_text)

    st.write(f'`{bits_encoded}` payload bits delivered at a bitrate of `{100*bits_encoded/(8*len(encoded_text)):.2f}`%')

    if st.sidebar.checkbox('Show German?'):
        st.markdown('## German Intermediary')
        st.write(otherlang_text)

if decode:
    example_encoded = "Three months after Japanese researchers invented transistors at American Bell Labs in 1947, they began research into transistors. Japanese companies used transistors and other electronic components in manufacturing radios, TVs, Sony Walkmans, video cassette recorders and computers. When Japanese companies regained 60% of the value value of Japanese goods under the 1985 Plaza agreement, they lost their competitive position and advanced into electronic value chains, focusing on exports of electronic components and capital goods to end-product manufacturers overseas."
    encoded_text = st.text_area("Cover + Payload", example_encoded)
    decoded_text = decode_long_text(cover_text,encoded_text,models,temperature=T,sampling_topk=topk)
    st.markdown('## Decoded Text')
    st.write(str(decoded_text)[1:]+'\n')