from ollama import chat
from embeddings import retrieve_chunks, create_models, embed_chunks
from chunking import chunk_pdf

MODEL_NAME = 'qwen3:4b-instruct-2507-q4_K_M'

SYSTEM_PROMPT = ("Tu réponds à des questions sur le règlement de l'Hyrox. "
                "Réponds en français, uniquement à partir des extraits fournis. " 
                "Termine ta réponse par la source : section, page et nom du document. "
                "Si la réponse n'est pas dans les extraits, dis-le et n'invente rien.")

def generate_answer(prompt : str) -> str :
    response = chat(
        model=MODEL_NAME,
        messages=[{'role': 'system', 'content' : SYSTEM_PROMPT},
                    {'role': 'user', 'content' : prompt}],
        think=False,
    )
    return response.message.content

def generate_prompt(chunks, question : str) -> str :
    content = ""
    for chunk in chunks :
        extrait = f"Page {chunk['page']} de source {chunk['source']}, {chunk['text']}\n-------------\n"
        content += extrait
    return content + f"Question : {question}"

if __name__ == "__main__":
    
    question = input("Écris la question que tu souhaites poser : ")
    while not question.strip() :
        question = input("Écris la question que tu souhaites poser : ")

    path_pdf = "../data/26_27_HYROX_RulebookDoubles_FR_210726.pdf"

    chunks = chunk_pdf(path_pdf)

    bi_encoder, reranker = create_models()

    chunk_embeddings = embed_chunks(chunks, bi_encoder)

    chunks = retrieve_chunks(question=question, chunks=chunks, bi_encoder=bi_encoder,
                                reranker=reranker, chunk_embeddings=chunk_embeddings)

    for chunk in chunks:
        print(f"Chunk : {chunk}\n") 
    print(generate_answer(prompt=generate_prompt(chunks=chunks, question=question)))