import sys
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from chunking import Chunk, chunk_pdf

TOP_K = 20
TOP_N = 8
BI_ENCODER_NAME = "intfloat/multilingual-e5-base"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

def embed_chunks (chunks : list[Chunk], model : SentenceTransformer) -> np.ndarray:
    chapters = []
    for c in chunks :
        if not c['description_table'] :
            chapters.append("passage: " + c['chapter'])
        else : 
            chapters.append("passage: " + c['description_table'])
    return model.encode(chapters, normalize_embeddings=True)

def select_best_per_section(chunks: list[Chunk], top_indices: list[int], top_n: int = TOP_N) -> list[Chunk]:
    selected, seen = [], set()

    for i in top_indices:
        if chunks[i]['section'] not in seen:
            selected.append(i)
            seen.add(chunks[i]['section'])
    if len(selected) < top_n :
        for i in top_indices:
            if i not in selected:
                selected.append(i)
            if len(selected) == top_n:
                break

    return selected


def rerank_scores(chunks : list[Chunk], question : str, top_indices : list[int], reranker : CrossEncoder) :
    pairs = [(question, chunks[i]['text']) for i in top_indices]
    return reranker.predict(pairs)

def create_models():
    return SentenceTransformer(BI_ENCODER_NAME), CrossEncoder(RERANKER_NAME)

def retrieve_chunks(question : str, chunks : list[Chunk], bi_encoder : SentenceTransformer, reranker: CrossEncoder, 
                    chunk_embeddings : np.ndarray) -> list[Chunk]: 
    question_embedding = bi_encoder.encode("query: " + question, normalize_embeddings=True)
    
    bi_scores = chunk_embeddings @ question_embedding
    top_indices = np.argsort(bi_scores)[::-1][:TOP_K]

    top_indices_per_section = select_best_per_section(chunks, top_indices)
    reranked = rerank_scores(chunks, question, top_indices_per_section, reranker)

    top_chunks = []
    for i in np.argsort(reranked)[::-1][:TOP_N] :
        top_chunks.append(chunks[top_indices[i]])

    return top_chunks


if __name__ == "__main__":
    
    question = input("Écrit la question que tu souhaites poser :")
    while not question.strip() :
        question = input("Écrit la question que tu souhaites poser :")

    path_pdf = "../data/26_27_HYROX_RulebookDoubles_FR_210726.pdf"

    chunks = chunk_pdf(path_pdf)

    bi_encoder, reranker = create_models()

    chunk_embeddings = embed_chunks(chunks, bi_encoder)

    question_embedding = bi_encoder.encode(question, normalize_embeddings=True)

    bi_scores = chunk_embeddings @ question_embedding

    top_index = np.argsort(bi_scores)[::-1][:TOP_K]

    new_scores = rerank_scores(chunks, question, top_index, reranker)

    for i in np.argsort(new_scores)[::-1][:3] :
        print(f"{new_scores[i]} Chunk {top_index[i]}: {chunks[top_index[i]]}")
        print()