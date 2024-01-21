import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai import SmartDatalake # for multiple files
#from pandasai import SmartDataframe # for single file
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.evaluation import load_evaluator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import StringDistance
import re

#set plot option for streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

#set maximum row,column size for uploaded files
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

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
st.info("Upload a file and start asking questions")

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
   # ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
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
# a simple regex expression for splitting words
def word_regex(sentence):
    samp = re.compile('[A-Za-z]+')
    word = samp.findall(sentence)
    return str(word)

#compare_strings func compares user prompt and dataframe columns
#returns column name
def compare_strings(user_string, df_string):
    user_strings= set(user_string.split())
    df_strings= set(df_string)
    common_words = user_strings.intersection(df_strings)
    return common_words#,user_strings , df_strings

#similarity score is mesaures similarity between user prompt and reference sentence
#reference sentence is defined as general info for a column
def similarity_score(prompt):
    # use jaro_evaluator as an example, there are different types of evaluator also for string distance
    #for instance levenshtein, indel
    score = jaro_evaluator.evaluate_strings(prediction=prompt, reference="column description")
    #score is a dictionary and define a list for getting score value
    v = list(score.values())
    return v[0]
jaro_evaluator = load_evaluator("string_distance", distance=StringDistance.JARO_WINKLER)


#upload a file from ui
uploaded_files = st.file_uploader("Please upload data file",
                                 type =list(file_formats.keys()),)# accept_multiple_files=True,

#check the uploaded file whether empty or not
if uploaded_files:
    dataframe = load_data(uploaded_files)
    if dataframe.empty:
        #if file is empty give a message
        st.write("The given data is empty, please upload a full file for asking questions")
    # if file is full, list first 3 rows
    else:
        st.write(dataframe.head(3))
    # give a general description of data
    if st.sidebar.button("Statistics"):
        if dataframe.empty:
            st.write("No description for an empty file")
    # if uploaded file is full, return description of data
        else:
            df_desc = dataframe.describe()
            st.write(df_desc)
    if st.sidebar.button("Information"):
        df_info=dataframe.info()
        st.write(df_info)
    if st.sidebar.button("Mean"):
        df_mean=dataframe.mean()
        st.write(df_mean)
    if st.sidebar.button("Median"):
        df_med=dataframe.median()
        st.write(df_med)
    if st.sidebar.button("Sample"):
        df_sample=dataframe.sample()
        st.write(df_sample)
    if st.sidebar.button("Correlation"):
        df_corr = dataframe.corr()
        st.write(df_corr)
    if st.sidebar.button("Drop All Duplicates"):
        df_drop_all=dataframe.drop_duplicates()
        st.write(df_drop_all)
    if st.sidebar.button("Drop All Null Values"):
        df_drop_null =dataframe.dropna(how='all')
        st.write(df_drop_null)


#define preffered llm
    llm = OpenAI(api_token='OPEN_AI_API_KEY',model="gpt-3.5-turbo-1106")

    # convert to SmartDatalake
    pandas_ai = SmartDatalake([dataframe],
                              config={"save_logs": True,
                                      "verbose": True,
                                      "enforce_privacy": True,
                                      "enable_cache": True,
                                      "use_error_correction_framework": True,
                                      "max_retries": 3,
                                      #"custom_prompts": {},
                                      "open_charts": True,
                                      "save_charts": False,
                                      "save_charts_path": "exports/charts",
                                      "custom_whitelisted_dependencies": [],
                                      "llm": llm,
                                      #"llm_options": null,
                                      "saved_dfs": []
                                    , "response_parser": StreamlitResponse
                                      #  share a custom sample head to the LLM  "custom_head": head_df

                                 }
                              )

#get question from user
    prompt =st.text_area("Please enter your question:")
    st.info("Type your question, if you want to get a chart you can specify the type of the chart")

   #check the prompt for questions intent with similarty_score model
   #figure out the intent is a general description for a file or not
   #set a threshold for similarity
    if len(prompt)>0.6:
        if similarity_score(prompt) > 0.5:
            user_column = str(prompt).lower()# lower is used for exact mathcing
            df_column = list(dataframe.columns)
            df_column = [x.lower() for x in df_column]#lower is used for exact mathcing
            #get the column which user request(mentioned in the prompt)
            col_name = compare_strings(user_column, df_column)
            col_name = [x.capitalize() for x in col_name]

    #show the results
            tab1, tab2 = st.tabs(["General Information for Column",
                                        "Distribuiton of Unique Value"
                                ]
                                       )

            for i in col_name:
                with tab1:
                    if dataframe[i].dtype =='int64':
                        #pandas provides sufficient info for general description of a column
                        #that's why static sentences are created for general info
                        st.write(
                            "For %s column, total number of rows is %s and the percentage of null values is %s . The number of different values in the column is  %s. The maximum value of the column is %s and minimum value is %s. The range is %s for the column.There is a graph for unique value counts at Distribuiton of Unique Value tab"
                            %( i, dataframe[i].shape[0] , dataframe[i].isnull().sum() / dataframe.shape[0] * 100 ,dataframe[i].nunique(),max(dataframe[i]),min(dataframe[i]),(dataframe[i].max()-dataframe[i].min())))

                    if dataframe[i].dtype == 'object':
                        st.write(
                            "For %s column, total number of rows is %s and the percentage of null values is %s . The number of different values in the column is  %s. There is a graph for unique value counts at Distribuiton of Unique Value tab"
                            % (i, dataframe[i].shape[0], dataframe[i].isnull().sum() / dataframe.shape[0] * 100,dataframe[i].nunique()))
                with tab2:
                    st.write("Unique value counts of column %s"%(i))
                    labels = dataframe[i].unique()
                    sizes =dataframe[i].value_counts()
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
                    st.pyplot()




    #get the request to generate an asnwer for the question
    if st.button("Generate Response"):
        # check uploaded file, if the file is empty give a message
        if dataframe.empty:
            st.write("No response for an empty file")
        #show the model's response as an answer
        else:
            if prompt:
                st.spinner("Calculating...")
                #st.write("Generating response, please wait...")
                gen_response = str(pandas_ai.chat(prompt))
                gen_code =pandas_ai.last_code_executed

                tab1, tab2,tab3 =st.tabs(["Generated Response",
                                          "Generated Chart",
                                          "Generated Code"]
                                         )
                with tab1:
                    if prompt:
                        st.write(str(pandas_ai.chat(prompt)))

                with tab2:
                    if 'pie' in prompt.lower():
                        fig = px.pie({prompt})
                        st.pyplot(plt.gcf())
                    if  'bar' in prompt.lower():
                        fig = px.bar({prompt})
                        st.pyplot(plt.gcf())
                    if 'bubble' in prompt.lower():
                        fig=px.scatter({prompt})
                        st.pyplot(plt.gcf())
                    if 'dot' in prompt.lower():
                        fig=px.scatter({prompt})
                        st.pyplot(plt.gcf())
                    if 'time series' in prompt.lower():
                        fig=px.line({prompt})
                        st.pyplot(plt.gcf())
                    else:
                        fig=px.histogram({prompt})
                        st.pyplot(plt.gcf())

                with tab3:
                    st.write(f"Generated code for the {prompt} :")
                    st.write(pandas_ai.last_code_executed)


            #if the question is empty give a message
            else:
                st.warning("Please ask a question")

