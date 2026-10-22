"""
app.py

Nyaya - Legal & Compliance Document Intelligence Pipeline
=========================================================
RAG pipeline:
retrieve -> assemble context -> answer -> cite
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

from retrieval import retrieve
from context import assemble_context
from otel_setup import get_tracer


# ---------------------------------------------------------
# ENVIRONMENT AND OPENTELEMETRY
# ---------------------------------------------------------

load_dotenv()

tracer = get_tracer("nyaya.app")


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

CHAT_MODEL = "gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-small"

COLLECTION_NAME = "nyaya_documents"
PERSIST_DIR = "./chroma_store"

MIN_CONFIDENCE = 0.05


SYSTEM_PROMPT = """
You are Nyaya, the NCS Telco+ Legal & Compliance Document Intelligence Assistant.

Tone: Precise, calm, and technical - the voice of a senior engineer.

Grounding rule: Only answer using the excerpts provided in each question.
Every factual claim in your answer MUST be followed by a citation in the
form (Source: <filename>). If the excerpts don't cover the question, say
so plainly and do not guess.
"""


# ---------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------

client = OpenAI()

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="Nyaya - Legal & Compliance Document Intelligence Pipeline"
)


print(vectorstore._collection.count())


@app.get("/")
def root():
    return {
        "message": "Nyaya is running. Try /ask?q=your+question&pattern=topk"
    }


# ---------------------------------------------------------
# ASK ENDPOINT
# ---------------------------------------------------------

@app.get("/ask")
def ask(q: str, pattern: str = "topk", k: int = 3):

    # Root span representing one complete Nyaya request
    with tracer.start_as_current_span("agent_turn") as root_span:

        root_span.set_attribute(
            "lab.question",
            q[:200]
        )

        root_span.set_attribute(
            "lab.retrieval_pattern",
            pattern
        )

        root_span.set_attribute(
            "lab.retrieval_k",
            k
        )


        # -------------------------------------------------
        # RETRIEVAL SPAN
        # -------------------------------------------------

        with tracer.start_as_current_span("retrieval") as retrieval_span:

            results = retrieve(
                vectorstore,
                q,
                pattern=pattern,
                k=k
            )

            retrieval_span.set_attribute(
                "retrieval.pattern",
                pattern
            )

            retrieval_span.set_attribute(
                "retrieval.k",
                k
            )

            retrieval_span.set_attribute(
                "retrieval.result_count",
                len(results)
            )


        # -------------------------------------------------
        # CONFIDENCE CHECK
        # -------------------------------------------------

        scored = [
            (doc, score)
            for doc, score in results
            if score is not None
        ]

        best_score = max(
            (score for _, score in scored),
            default=None
        )

        if best_score is not None:
            root_span.set_attribute(
                "lab.best_relevance_score",
                float(best_score)
            )


        # Refuse to answer if retrieval did not return
        # sufficiently relevant information.
        if (
            not results
            or (
                best_score is not None
                and best_score < MIN_CONFIDENCE
            )
        ):

            root_span.set_attribute(
                "lab.answer_status",
                "insufficient_context"
            )

            return {
                "question": q,
                "pattern": pattern,
                "retrieved_chunks": [],
                "answer": (
                    "I don't know - none of the ingested documents "
                    "cover this question confidently enough for me "
                    "to answer."
                ),
                "sources": [],
            }


        # -------------------------------------------------
        # CONTEXT ASSEMBLY SPAN
        # -------------------------------------------------

        with tracer.start_as_current_span(
            "context_assembly"
        ) as context_span:

            context_text, used_results = assemble_context(
                results
            )

            context_span.set_attribute(
                "context.chunk_count",
                len(used_results)
            )

            context_span.set_attribute(
                "context.length",
                len(context_text)
            )


        # -------------------------------------------------
        # PREPARE RETRIEVED CHUNKS
        # -------------------------------------------------

        retrieved_chunks = [
            {
                "source": doc.metadata.get(
                    "source",
                    "unknown"
                ),

                "metadata": {
                    k2: v
                    for k2, v in doc.metadata.items()
                    if k2 != "source"
                },

                "relevance_score": (
                    round(score, 4)
                    if score is not None
                    else None
                ),

                "text_preview": (
                    doc.page_content[:150]
                    + (
                        "..."
                        if len(doc.page_content) > 150
                        else ""
                    )
                ),
            }

            for doc, score in used_results
        ]


        # -------------------------------------------------
        # BUILD PROMPT
        # -------------------------------------------------

        user_prompt = (
            f"Question: {q}\n\n"
            f"Retrieved excerpts:\n{context_text}\n\n"
            "Answer using only the excerpts above. "
            "Cite each fact as (Source: <filename>)."
        )


        # -------------------------------------------------
        # MODEL CALL SPAN
        # -------------------------------------------------

        with tracer.start_as_current_span(
            "model_call"
        ) as model_span:

            model_span.set_attribute(
                "gen_ai.model",
                CHAT_MODEL
            )

            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )


        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        answer = response.choices[0].message.content

        root_span.set_attribute(
            "lab.answer_status",
            "success"
        )

        root_span.set_attribute(
            "lab.source_count",
            len(
                set(
                    doc.metadata.get(
                        "source",
                        "unknown"
                    )
                    for doc, _ in used_results
                )
            )
        )

        return {
            "question": q,
            "pattern": pattern,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "sources": sorted(
                set(
                    doc.metadata.get(
                        "source",
                        "unknown"
                    )
                    for doc, _ in used_results
                )
            ),
        }


# ---------------------------------------------------------
# LOCAL START
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )