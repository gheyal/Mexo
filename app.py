import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re

st.set_page_config(page_title="Mexo", page_icon="🤖")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gheyal/Mexo")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, "gheyal/Mexo")
    return tokenizer, model

tokenizer, model = load_model()

def try_calculate(query):
    q = query.lower().replace('times', '*').replace('multiplied by', '*').replace('plus', '+').replace('minus', '-').replace('divided by', '/')
    match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*/x])\s*(\d+(?:\.\d+)?)', q)
    if match:
        a, op, b = match.groups()
        a, b = float(a), float(b)
        op = '*' if op == 'x' else op
        try:
            return f"{a:g} {op} {b:g} = {eval(f'{a}{op}{b}'):g}"
        except Exception:
            return None
    return None

def clean_answer(text):
    for i, ch in enumerate(text):
        if ch in '.!?':
            return text[:i+1].strip()
    return text.strip()

def answer(query):
    calc = try_calculate(query)
    if calc:
        return calc
    prompt = "Answer the following question directly and concisely.\n\nQuestion: " + query + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.3,
        eos_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    result = full_output.split("Answer:")[-1].strip() if "Answer:" in full_output else full_output
    return clean_answer(result)

st.title("🤖 Mexo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = answer(prompt)
            st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})