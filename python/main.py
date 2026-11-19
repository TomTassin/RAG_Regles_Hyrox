from embeddings import retrieve_chunks, create_models, embed_chunks
from chunking import chunk_pdf
from generation import generate_answer, generate_prompt

question = input("Écris la question que tu souhaites poser : ")
while not question.strip() :
    question = input("Écris la question que tu souhaites poser : ")

path_pdf = "../data/26_27_HYROX_RulebookDoubles_FR_210726.pdf"

chunks = chunk_pdf(path_pdf)

bi_encoder, reranker = create_models()

chunk_embeddings = embed_chunks(chunks, bi_encoder)

chunks = retrieve_chunks(question=question, chunks=chunks, bi_encoder=bi_encoder,
                                reranker=reranker, chunk_embeddings=chunk_embeddings)

print(generate_answer(prompt=generate_prompt(chunks=chunks, question=question)))