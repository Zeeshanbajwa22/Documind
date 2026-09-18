from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

# Let's see what we got
print(f"Number of pages loaded: {len(pages)}")
print("---")
print("First page content preview:")
print(pages[0].page_content[:500])  # first 500 characters
print("---")
print("Metadata of first page:")
print(pages[0].metadata)


from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the pages into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # each chunk ~1000 characters
    chunk_overlap=200,    # 200 characters overlap between chunks
)

chunks = text_splitter.split_documents(pages)

print(f"Number of chunks created: {len(chunks)}")
print("---")
print("First chunk content:")
print(chunks[0].page_content)
print("---")
print("First chunk metadata:")
print(chunks[0].metadata)