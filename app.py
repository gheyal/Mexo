import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(
    page_title="Mexo",
    page_icon="🤖"
)

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER = "gheyal/Mexo-1.7B"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def answer(question):
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return result.strip()


st.title("🤖 Mexo")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input("Ask something..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mexo is thinking..."):
            reply = answer(prompt)

        st.write(reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })