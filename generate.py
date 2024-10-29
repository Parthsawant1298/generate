import os
import streamlit as st
import pandas as pd
import io
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

# Set cache directory explicitly
os.environ['TRANSFORMERS_CACHE'] = r"C:\Users\parth sawant\.cache"
os.environ['HF_HOME'] = r"C:\Users\parth sawant\.cache"

# Disable TensorFlow oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize session state for model and tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        
        # Load tokenizer with explicit cache directory
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=r"C:\Users\parth sawant\.cache"
        )
        
        # Check for GPU availability and set appropriate device
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
        # Load model with appropriate configuration and cache directory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            cache_dir=r"C:\Users\parth sawant\.cache"
        )
        
        if device == "cpu":
            model = model.to(device)
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Hugging Face login
HF_TOKEN = "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"  # Replace with your actual token
login(HF_TOKEN)

# Function to generate dataset based on a text prompt using Llama
def generate_dataset_from_text(text, model, tokenizer):
    if not model or not tokenizer:
        raise ValueError("Model or tokenizer not properly loaded")
    
    # Create a chat prompt format that Llama-2-chat understands
    prompt = f"""[INST]Generate a dataset in CSV format based on the following text. 
    Return only the CSV data with a header row and data rows. Each row must have the same number of columns:
    
    {text}[/INST]"""
    
    try:
        # Encode the prompt and move to the same device as the model
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generate response tokens with proper parameters for Llama-2
        with torch.no_grad():
            response_token_ids = model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                repetition_penalty=1.2
            )
        
        # Decode generated tokens to text
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        
        # Extract only the CSV part from the response
        if "[/INST]" in generated_text:
            generated_text = generated_text.split("[/INST]")[1].strip()
        
        return generated_text
    
    except Exception as e:
        raise Exception(f"Error in text generation: {str(e)}")

# Function to clean CSV data for parsing
def clean_csv_data(csv_data):
    # Split into lines and remove empty lines
    lines = csv_data.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    # Ensure consistent number of columns
    if cleaned_lines:
        first_line_cols = len(cleaned_lines[0].split(','))
        cleaned_lines = [
            line for line in cleaned_lines 
            if len(line.split(',')) == first_line_cols
        ]
    
    return "\n".join(cleaned_lines)

# Function to validate CSV format
def validate_csv_format(csv_data):
    lines = csv_data.splitlines()
    if not lines:
        return False, "The CSV data is empty."
    
    header_columns = len(lines[0].split(','))
    if header_columns == 0:
        return False, "No columns found in the header."
    
    errors = []
    for i, line in enumerate(lines[1:], start=2):
        cols = len(line.split(','))
        if cols != header_columns:
            errors.append(f"Line {i} has {cols} columns (expected {header_columns}).")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, None

# Function to save the generated dataset
def save_dataset_to_folder(csv_data, file_name="generated_dataset.csv"):
    dataset_folder = os.path.join(os.getcwd(), "datasets")
    os.makedirs(dataset_folder, exist_ok=True)
    
    file_path = os.path.join(dataset_folder, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(csv_data)
    
    return file_path

# Streamlit App
st.set_page_config(page_title="Dataset Generator", layout="wide")
st.title("🗂️ Dataset Generator from Text using Llama")

# Add a status indicator for model loading
st.sidebar.title("Model Status")
model_status = st.sidebar.empty()

# Load model and tokenizer
if st.session_state.model is None or st.session_state.tokenizer is None:
    model_status.warning("Loading Llama model... This may take a few minutes...")
    st.session_state.model, st.session_state.tokenizer = load_model_and_tokenizer()
    if st.session_state.model is not None:
        model_status.success("Model loaded successfully!")
    else:
        model_status.error("Failed to load model")

if st.session_state.model is None or st.session_state.tokenizer is None:
    st.error("Failed to load the model. Please check your cache directory and try again.")
else:
    # Input field for text prompt
    text_prompt = st.text_area(
        "Enter a prompt to generate a dataset:",
        height=150,
        help="Describe the kind of dataset you want to generate"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        generate_button = st.button("Generate Dataset", use_container_width=True)
    
    if generate_button:
        if text_prompt:
            try:
                progress_bar = st.progress(0)
                st.info("Generating dataset... Please wait.")
                
                # Generate dataset from the text prompt
                csv_data = generate_dataset_from_text(
                    text_prompt,
                    st.session_state.model,
                    st.session_state.tokenizer
                )
                progress_bar.progress(50)
                
                # Clean and validate the CSV data
                cleaned_csv_data = clean_csv_data(csv_data)
                is_valid, error_message = validate_csv_format(cleaned_csv_data)
                progress_bar.progress(75)
                
                if not is_valid:
                    st.error(f"Generated CSV data is incorrectly formatted:\n{error_message}")
                else:
                    try:
                        df = pd.read_csv(io.StringIO(cleaned_csv_data))
                        st.subheader("Preview of Generated Dataset:")
                        st.dataframe(df, use_container_width=True)

                        # Save and provide download option
                        saved_file_path = save_dataset_to_folder(cleaned_csv_data)
                        
                        st.download_button(
                            label="Download CSV",
                            data=cleaned_csv_data,
                            file_name='generated_dataset.csv',
                            mime='text/csv',
                            key='download_button'
                        )
                        
                        st.success(f"CSV file saved to: {saved_file_path}")
                        progress_bar.progress(100)
                        
                    except Exception as e:
                        st.error(f"Error processing CSV: {str(e)}")
                
            except Exception as e:
                st.error(f"Error generating dataset: {str(e)}")
        else:
            st.warning("Please provide a text prompt.")

    # Add some usage tips
    with st.expander("Usage Tips"):
        st.markdown("""
        ### Tips for better results:
        1. Be specific about the type of data you want
        2. Mention the desired columns and their types
        3. Specify any constraints or patterns in the data
        4. Include the number of rows you want (default is around 5-10 rows)
        
        ### Example prompt:
        "Generate a dataset of customer information with columns for name, age, email, and purchase amount. Include 5 rows of realistic data."
        """)
