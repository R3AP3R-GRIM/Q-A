import streamlit as st
from transformers import pipeline
import easyocr
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
from transformers import LongformerTokenizerFast
import torch
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

text=st.text_area('Context',height=350)
question = st.text_input('Questions?')
    if(st.button('Submit')):
        def longformer(text, question):
            encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            start_scores, end_scores = model(input_ids, attention_mask=attention_mask,return_dict=False)
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores)+1]
            answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
            return answer
        ans= (longformer(text, question))
        st.subheader('Question-')
        st.write('Q.' + question)
        st.subheader("Answer:")
        st.write(ans)
