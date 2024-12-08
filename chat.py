import streamlit as st
import torch
import time
import os
import filelock

global_lock = filelock.FileLock(".opengrok.lock")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

st.set_page_config(page_title="OpenGrok-2", page_icon="🤖", )
hide_streamlit_style = """
<style>#root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("💬 OpenGrok-2")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_id = 'LLM-Research/Meta-Llama-3-8B-Instruct'
models_dir = './models'
model_path = f"{models_dir}/model/{model_id.replace('.', '___')}"
lora_dir = f"./models/lora/{model_id}"
torch_dtype = torch.bfloat16

@st.cache_resource
def init_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from peft import LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.bfloat16)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, 
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = PeftModel.from_pretrained(model, model_id=lora_dir, config=config)
    return tokenizer, model


def bulid_input(prompt, history=[]):
    """
    This function constructs the input prompt for the LLM model. It formats the user's input and the conversation history
    into a string that can be fed into the model.

    Parameters:
    - prompt (str): The user's input prompt.
    - history (list, optional): A list of dictionaries representing the conversation history. Each dictionary contains the 'role' and 'content' of a message.

    Returns:
    - str: The formatted input prompt for the LLM model.
    """
    system_format = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format = '<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n'
    history.append({'role': 'user', 'content': prompt})
    prompt_str = ''
    for item in history:
        if item['role'] == 'user':
            prompt_str += user_format.format(content=item['content'])
        else:
            prompt_str += assistant_format.format(content=item['content'])
    return prompt_str + '<|start_header_id|>assistant<|end_header_id|>\n\n'



if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

tokenizer, model = init_model()

if prompt := st.chat_input(placeholder="Hi, I'm OpenGrok."):
    st.chat_message("user").write(prompt)

    input_str = bulid_input(prompt=prompt, history=st.session_state["messages"])
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(device)

    get_lock = False
    try:
        global_lock.acquire(timeout=60)
        get_lock = True
    except filelock.Timeout:
        st.session_state.messages.append({"role": "assistant", "content": "Too many requests, please try again."})
        st.chat_message("assistant").write("Too many requests, please try again.")

    if get_lock:
        try:
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,
                top_p=0.5,
                temperature=0.5,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.encode('<|eot_id|>')[0]
            )
            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs)
            response = (response.strip()
                        .replace('<|eot_id|>', "")
                        .replace('<|start_header_id|>assistant<|end_header_id|>', '')
                        .replace('<|end_of_text|>', '').strip())
    
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        finally:
            global_lock.release()

