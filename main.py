from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')

video_id = input("Enter the video_ide : ")

try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["hi",'en']
    )

    transcript = " ".join(snippet.text for snippet in transcript_list)

    # print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled.")
    exit()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
chunks = splitter.create_documents([transcript])

# print(len(chunks))

vector_store = FAISS.from_documents(chunks,embedding=embeddings)

# print(vector_store.index_to_docstore_id)

#2 Retrieval

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

# results = retriever.invoke('What is Anchor technique ?')

# print(results)

#3 Augumentation LLM

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

prompt = PromptTemplate(
    template='''
        You are a helpful assitant,
        Answer ONLY from the provided transcript context.
        If the context is insufficienet, just say you don't know.

        {context}
        Question: {question}
        ''',
        input_variables=['context','question']
)

while(1):
    question = input("Enter your question regarding the video(enter exit) : ")
    if(question == 'exit'):
        break
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({'context': context_text, 'question': question})

    #4 Generation 

    answer = llm.invoke(final_prompt)

    print("\n\n",answer.content)