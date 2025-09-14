import PyPDF2
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

pdf_path = "Regeln.pdf"

def main():
    load_dotenv()
    
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n", 
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len)
        
        chunks = text_splitter.split_text(text)

        print(f"Number of chunks: {len(chunks)}")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

        documents = []

        for chunk in chunks:
           document = Document(chunk, metadata={"source": pdf_path})
           documents.append(document)

        knowledge_base = FAISS.from_documents(documents, embeddings)

        model = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

        template = """
        Du bist experte in den Regeln des Österreichischen Fussballbundes (ÖFB).

        Hier sind einige relevante Auszüge aus dem Regelwerk: {rules}

        Hier ist die Frage die du beantworten sollst: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | model

        while True:
            print("\n\n-------------------------------")
            question = input("Ask your question (q to quit): ")
            print("\n\n")
            if question == "q":
                break

            retriever = knowledge_base.as_retriever()

            # docs = knwledge_base.similarity_search(question)         
            rules = retriever.invoke(question)           


            results = chain.invoke({"rules": rules, "question": question})
            print(results)

if __name__ == '__main__':
    main()