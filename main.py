import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai import SmartDatalake # for multiple files
#from pandasai import SmartDataframe # for single file
from pandasai.llm.openai import OpenAI

#set maximum row size for uploaded files
pd.options.display.max_rows = 999999

# load environment variables
load_dotenv()

# check open_ai_api_key in env, if it is not defined as a variable
#it can be added manually in the code below

if os.environ.get("OPEN_AI_API_KEY") is None or os.environ.get("OPEN_AI_API_KEY") =="":
    print("open_ai_api_key is not set as environment variable")
else:
    print("Open AI API Key is set")

#get open_ai_api_key
OPEN_AI_API_KEY= os.environ.get("OPEN_AI_API_KEY")

# set tittle for Streamlit UI
st.title("Explore Data Using Natural Language")

#set formats of files allowed to upload
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

#define a function for file formats
#check the file format among the list above
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    #if file format does not match give message
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

#upload file from ui
uploaded_file = st.file_uploader("Please upload data file",
                                 type =list(file_formats.keys()),)

#check the uploaded file whether empty or not
if uploaded_file:
    df = load_data(uploaded_file)
    # if file is empty give a message
    if df.empty:
        st.write("Uploaded file is empty, please upload a full file for analyzing")
    # if file is full, list first 3 rows
    else:
        st.write(df.head(3))

    #give a general description of data
    if st.button("Statistics of Data"):
        # if uploaded file is empty, give a message
        if df.empty:
            st.write("No description for an empty file")
        #if uploaded dile is full, return description of data
        else:
            df_desc = df.describe()
            st.write(df_desc)

#define preffered llm
    llm = OpenAI(api_token='OPEN_AI_API_KEY',model="gpt-3.5-turbo")

#convert to SmartDatalake
    pandas_ai = SmartDatalake([df],
                              config={"llm": llm, "verbose": True,"enforce_privacy":True, "verbose":True
                                      # if there are custom prompts add as a config
                                      #"custom_prompts": {
                                      #    "generate_python_code": MyCustomPrompt(
                                      #        my_custom_value="my custom value")
                                      }
                              )

#get question from user
    prompt =st.text_area("Please enter your question:")

#get the request to generate an asnwer for the question
    if st.button("Generate Response"):
        # check uploaded file, if the file is empty give a message
        if df.empty:
            st.write("No response for an empty file")
        #show the model's response as an answer
        else:
            if prompt:
                st.write("Generating response, please wait...")
                st.write(pandas_ai.chat(prompt))
            #if the question is empty give a message
            else:
                st.warning("Please enter a question")







