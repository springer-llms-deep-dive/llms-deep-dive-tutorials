# Tutorial: Building your own retrieval-augmented generation system

## Overview

Retrieval-augmented generation is a promising avenue for combining the
semantic understanding capabilities of LLMs with the factual accuracy of
direct source materials. In this chapter, we have discussed many aspects
of these systems, from basic approaches to optimization to enhancements
to the core RAG functionality and methods for evaluating RAG
performance. This section will present a practical, hands-on example of
building and augmenting a RAG system using a popular open-source RAG
application.

In recent years, a number of such open-source libraries have been
developed and released. These RAG libraries present high-level functions
for implementing many different RAG approaches and allow great
customization for constructing one's own system. For this tutorial, we
will use LlamaIndex to build a RAG system,
experiment with a few of the many tunable parameters, and evaluate the
system's performance.

**Goals:**

-   Demonstrate how to set up a basic RAG application with low effort
    using LlamaIndex.

-   Explore the wide range of possibilities for customizing and
    improving a RAG application.

-   Evaluate context relevance, answer relevance, and groundedness for a
    RAG application.

### Tools and libraries

-   **LlamaIndex**: LlamaIndex is a data framework for LLM applications,
    including chatbot agents and RAG systems, with both open-source and
    enterprise offerings. LlamaIndex handles document parsing, indexing,
    search, and generation using an extensive catalog of modules that
    can be easily incorporated into a single RAG framework. Integrations
    with Hugging Face allow for great customization in the choice of
    embedding and generation models. (<https://www.llamaindex.ai/>)

-   **BAAI/bge-small-en-v1.5**: The small English variant of the Beijing
    Academy of Artificial Intelligence's (BAAI) line of text-embedding
    models. This model is highly performant in text-similarity tasks,
    yet is small enough (3̃3.4M parameters) to fine-tune easily. (<https://huggingface.co/BAAI/bge-small-en-v1.5>)

-   **OpenAI ChatGPT**: Throughout the tutorial, we will be using the
    `gpt-3.5-turbo` and `gpt-4` models from OpenAI as our generators.
    They will also provide a comparison of the output of our RAG
    systems.

### Datasets

-   **OpenAI Terms & Policies**: For our document corpus, we will be
    using the OpenAI terms and policies, taken from
    <https://openai.com/policies>, as they appeared in late January 2024.

## Indexing

The first step is to load each document from the OpenAI terms &
conditions into a LlamaIndex `Document` object:

``` {.python language="Python" caption="\"Loading documents\""}
import os
from llama_index.core import Document

directory = './openai_tos/'
doc_names = os.listdir(directory)
documents = []

for i, doc_name in enumerate(doc_names):
    document = open(os.path.join(directory, doc_name)).read()
    d = Document(text=document,
        metadata = {
        "file": doc_name,
        "name": doc_name.split('_')[1].split('.')[0].replace('-',' ')
        })
    documents.append(d)
```

Each `Document` object takes a string of text, read from the files, and
a metadata dictionary which includes additional terms that can be used
for index filtering or keyword searches. Next we choose a chunking
strategy and an embedding model to generate our vector index.

``` {.python language="Python" caption="\"Model parameters\""}
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

text_chunker = SentenceSplitter(chunk_size=128, chunk_overlap=8)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

This basic splitting strategy attempts to create chunks roughly 128
tokens long but with a preference for not splitting sentences between
multiple chunks. We are using the small BGE English model produced by
BAAI, a highly performant text-similarity embedding model that is small
enough to fine-tune easily. For generation, we choose `gpt-3.5-turbo`.
We then use the text chunker to split our documents into notes and
create a vector index:

``` {.python language="Python" caption="\"Generate vector-index\""}
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex 

# Model parameters setup
text_chunker = SentenceSplitter(chunk_size=128, chunk_overlap=8)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Generating vector-index
nodes = text_chunker.get_nodes_from_documents(documents)
index = VectorStoreIndex(
    nodes,
    embed_model=embed_model,
    show_progress=True
)
index.storage_context.persist(persist_dir = './bge-small_openai-tos_vectors/')
```

## Basic querying

After this has finished, we can easily begin querying the RAG database
using `gpt-3.5-turbo` as the generator LLM.

``` {.python language="Python" caption="\"Simple query engine\""}
llm_gpt = OpenAI(model='gpt-3.5-turbo', temperature=0.0)
query_engine = index.as_query_engine(llm=llm_gpt)
```

With this initial RAG application, we can ask a question and compare
results to base ChatGPT's response:

``` 
    query = 'Who owns the content created by OpenAI programs?'

    GPT response:
    As of March 1st, 2023, OpenAI retains ownership of the content created by
    its AI models, including those generated by the ChatGPT model used by
    Assistant. However, OpenAI provides users with a license to use the
    content generated by the models. ...

    RAG response:
    The content created by OpenAI programs is owned by the user who provided
    the input and the output is owned by the user as well.
```

These responses give the opposite answers, and the RAG model is correct.
We can be confident in this by examining the nodes that RAG retrieved to
formulate its answer:

``` {.python language="Python" caption="\"First query documents\""}
for node in rag.source_nodes:
    print(node)

# Output
Node ID: 9d8a4293-dbad-4e98-88b6-e5f09bcb2315
Text: Ownership of Content. As between you and OpenAI, and to the
extent permitted by applicable law, you (a) retain your ownership
rights in Input and (b) own the Output. We hereby assign to you all
our right, title, and interest, if any, in and to Output.   Similarity
of Content. Due to the nature of our Services and artificial
intelligence generally...
Score:  0.763

Node ID: 85b44478-2664-4b65-837e-2cf50d844ca5
Text: Content co-authored with the OpenAI API Creators who wish to
publish their first-party written content (e.g., a book, compendium of
short stories) created in part with the OpenAI API are permitted to do
so under the following conditions:  The published content is
attributed to your name or company. The role of AI in formulating the
content is clearly...
Score:  0.761
```

By default, our llama_index system has collected two nodes and created a
response using `gpt-3.5-turbo` to distill relevant information. While it
came to the correct conclusion, it is not an especially detailed
response, so let us try some optimizations.

## Improved querying

First, we'll increase the number of chunks RAG uses from two to five by
making an explicit retrieval object.

``` {.python language="Python" caption="\"Improved retriever\""}
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)
```

We can also customize the generator prompt we saw above, which gives
short answers. A more verbose response that reports all of the relevant
details given in the retrieved documents would be helpful:

``` {.python language="Python" caption="\"Improved generator prompt\""}
# Assuming necessary imports for PromptTemplate
qa_prompt_tmpl = PromptTemplate(
    "You are an expert Q&A system that is trusted around the world. Always " \
    "answer the query using the provided context information, and not prior " \
    "knowledge. Context information is below.\n" \
    "---------------------\n{context_str}\n---------------------\n" \
    "Given the context information and not prior knowledge answer the query.\n" \
    "Query: {query_str}\n" \
    "Answer: "
)
```

This is a modification of the default prompt that removed some wording
that empirically was creating terse responses. Finally, I will change
the summarization method. By default, LlamaIndex summarizes each
individual document to create a shorter total input. I will instead just
have it include every retrieved chunk in a list. Finally, I will use
GPT-4 instead of GPT-3.5.

``` {.python language="Python" caption="\"Construct improved query engine\""}
# Assuming necessary imports and previous definitions for RetrieverQueryEngine, get_response_synthesizer, ResponseMode, and qa_prompt_tmpl

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE
)

custom_query_engine = RetrieverQueryEngine.from_args(
    retriever,
    response_synthesizer=response_synthesizer,
)

custom_query_engine.update_prompts(
     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

llm_gpt = OpenAI(model='gpt-4', temperature=0.0)
```

With this improved engine, we can rerun the same query.

``` {caption="\"Improved first query\""}
query = 'Who owns the content created by OpenAI programs?'

RAG response:
As per the provided context information, the ownership of content created
by OpenAI programs is as follows:
- Users retain ownership rights in the input they provide to OpenAI services.
- Users own any output they rightfully receive from OpenAI services, to the
  extent permitted by law.
- OpenAI assigns to users all their right, title, and interest, if any, in and
  to the output generated by OpenAI programs.
```

By increasing the number of retrieved documents and changing the
summarization strategy, our generator was given more information
in-context to formulate into an answer. In addition, the improved prompt
and better generator model have created a more interesting structure to
the response. We can already see that applying just a few of the
optimizations discussed in Sects. 7.3 and 7.4 creates a much better RAG system.

## Response reranking

RAG systems can also be augmented with a reranker, As described in
Sect. 7.4.3.1, rerankers embed the retrieved documents a
second time with a more powerful LLM, and rerank them according to
semantic similarity in the richer embedding space. For this test, let's
ask a slightly more complicated question and see how a reranker can be
advantageous.

``` {caption="\"Second query\""}
query = 'Will OpenAI give me any sort of reward if I find an issue with their
software?'
```

This is a reference to the bug bounty program described in OpenAI's
Coordinated Vulnerability Disclosure Policy. First, we query the
enhanced system described above:

``` {caption="\"Second query response\""}
RAG response:
No, OpenAI does not mention providing rewards for finding issues with their
software in the provided context information.
```

This system failed to notice the bug bounty program because the
difference in language between the query and text in the document is too
great. The top returned document is related to undesirable output but
not the bounty

``` {caption="\"Second query documents\""}
Our Services may provide incomplete, incorrect, or offensive Output that does not
represent OpenAI’s views. If Output references any third-party products or services,
it doesn’t mean the third-party endorses or is affiliated with OpenAI. ...
```

Let's ask the same question to a system with a reranker included. For
this system, we initially return the top 20 documents and then rerank
them using `gpt-3.5-turbo` embeddings, returning the top 5. We add a
reranker node processor to our query engine:

``` {.python language="Python" caption="\"Re-ranker\""}
custom_query_engine_rerank = RetrieverQueryEngine.from_args(
    retriever=VectorIndexRetriever(index=index, similarity_top_k=20),
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        LLMRerank(choice_batch_size=5, top_n=5),
    ]
)
```

With this engine, we get the following response based on the following
top document:

``` {caption="\"Second query reranked response and top document\""}
RAG response:
Yes, OpenAI offers a Bug Bounty Program as a way to recognize and reward security
researchers who contribute to keeping their technology and company secure.

Top result:
...
Bug bounty: The OpenAI Bug Bounty Program is a way for us to recognize and
reward the valuable insights of security researchers who contribute to keeping
our technology and company secure.
```

The superior embeddings could correctly associate a bug bounty program
with a reward. Looking back at the document rankings by the original
embedder, this document was the #12 match to our query and thus was not
captured by the improved query engine. After being reranked by the
larger embedding model, this document catapulted to #1.

## Embedding fine-tuning

Let's try a separate approach to augmenting our RAG model. We find that
there are consistently accuracy issues in our improved query engine when
the questions are somewhat vague. Consider the following:

``` {caption="\"Third query\""}
query = 'How does ChatGPT deal with inappropriate questions?'

RAG response:
ChatGPT ensures that automated systems disclose to people that they are
interacting with AI, unless it's obvious from the context. Additionally,
ChatGPT does not allow the building of tools that may be inappropriate for
minors, including sexually explicit or suggestive content, unless it is
created for scientific or educational purposes.
```

The expectation here is that the model would understand that we are
asking about material filtered out by their harmfulness filters. It
alludes to this in the second sentence, but the first sentence is
entirely on a different subject, suggesting that the top returned
documents were irrelevant. Let's see if we can fine-tune our embeddings
and improve performance.

To do this, we must collect a sample of queries matched to a text
segment that provides the answer. Instead of creating these by hand,
which would be quite taxing, we can automate the process using LLMs.
First, we create a train/validation split among all the segments we
generated by splitting up the OpenAI TOS documents. We pass both
datasets to an LLM and have it generate questions based on the content
of the nodes. When done, we can fine-tune the model to select the source
text segment based on each query. LlamaIndex has a function to
accomplish this:

``` {.python language="Python" caption="\"Q&A pairs generation\""}
from llama_index.finetuning import generate_qa_embedding_pairs
from sklearn.model_selection import train_test_split
import pandas as pd

train_nodes, val_nodes = train_test_split(
    pd.Series(nodes), test_size=0.20, random_state=7
)
train_dataset = generate_qa_embedding_pairs(
    train_nodes, llm=OpenAI(model='gpt-3.5-turbo'),
)
val_dataset = generate_qa_embedding_pairs(
    val_nodes, llm=OpenAI(model='gpt-3.5-turbo'),
)
```

Here is an example text segment and the question that was automatically
generated from it:

``` {caption="\"Q&A pairs example\""}
Sample text:
------------------
If you notice that ChatGPT output contains factually inaccurate information
about you and you would like us to correct the inaccuracy, you may submit a
correction request through privacy.openai.com or to dsar@openai.com. Given the
technical complexity of how our models work, we may not be able to correct the
inaccuracy in every instance. In that case, you may request that we remove your
Personal Information from ChatGPT’s output by filling out this form.

Query based on text:
---------------------
How can individuals request corrections for factually inaccurate information
about themselves in ChatGPT output?
```

Based on a large sample of such Q&A pairs, we can fine-tune the model
using sentence-transformers:

``` {.python language="Python" caption="\"Model fine-tuning\""}
from llama_index.finetuning import SentenceTransformersFinetuneEngine

finetune_engine = SentenceTransformersFinetuneEngine(
    model_id="BAAI/bge-small-en-v1.5",
    dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=3,
    model_output_path="./bge-small_openai-tos-finetuned_model"
)

finetune_engine.finetune()
```

This procedure updates the weights of `bge-small-en-v1.5` to encourage
the model to return the correct document for each query generated in the
above procedure. Next, we will pass every query from the validation
hold-out set to both the original and the new embedding model and
determine how often they retrieve the document that the message was
generated from:

``` {.python language="Python" caption="\"Fine-tuning evaluation\""}
# Python code for evaluating fine-tuning
def evaluate_st(dataset, model_id, name="evaluation"):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)
    model = SentenceTransformer(model_id)
    return evaluator(model, output_path=output_path)

# Output of evaluations
evaluate_st(val_dataset, "BAAI/bge-small-en-v1.5")
# Output: 0.9361111111111111

evaluate_st(val_dataset, "bge-small_openai-tos-finetuned_model")
# Output: 0.9551767676767677
```

Retrieval with the original model returns the correct node 93.6% of the
time, and retrieval with the fine-tuned model returns the correct node
95.5% of the time. This is a modest but real increase -- the small size
is largely due to the original model already working quite well. In
other cases, improvement can be around 10%. Now that we have the
fine-tuned model, we re-embed the documents and create a new query
engine:

``` {caption="\"Third query fine-tuned\""}
query = 'How does ChatGPT deal with inappropriate questions?'

Fine-tuned RAG response:
ChatGPT does not support the generation or promotion of inappropriate content,
including sexually explicit or suggestive content, disinformation, misinformation,
false online engagement, impersonation, academic dishonesty, or misleading others
about the purpose of the GPT. Users are encouraged to report any inappropriate
content generated by ChatGPT and may request corrections or removal of inaccurate
information through the provided channels in the privacy policy.
```

This is a rather more comprehensive and useful answer, driven by the
fact that the top documents are more relevant.

## RAG Evaluation

To close out the tutorial, let's evaluate the quality of our fine-tuned,
reranking-enabled RAG applications. There are many different approaches
to evaluation, but we will consider here only the three quality metrics
given in Sect. 7.5.1:

They are:

1.  Context Relevance: Is the retrieved context relevant to the query?

2.  Answer Relevance: Is the generated answer relevant to the query?

3.  Answer Faithfulness/Groundedness: Is the generated answer supported
    by the retrieved context?

This triad ensures that the query finds useful documents, that the
generated response is faithful to the documents, and that the generated
response answers the original query.

Looking first at context relevance, we can use the LlamaIndex
RetrieverEvaluator function. This function takes in a query, finds the
most similar documents, and compares those documents against an
expectation. We can use the validation Q&A dataset created for
fine-tuning for this evaluation -- we pass it to each generated query
and expect to get the document it was created from.

``` {.python language="Python" caption="\"Context relevance function\""}
from llama_index.core.evaluation import RetrieverEvaluator

def run_context_relevance_eval(index, queries, expected_ids):
    retriever = index.as_retriever(similarity_top_k=2)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    context_eval_results = []
    for cid, q in zip(expected_ids, queries):
        context_relev_eval = retriever_evaluator.evaluate(
            query=q, expected_ids=[cid,]
        )
        context_eval_results.append(context_relev_eval)
```

Here we ask the function for two metrics: \"mrr\", or mean reciprocal
rank (= 1 divided by the rank of the correct document), and
\"hit_rate\", which is equal to 1 if the correct document is among the
top 5 returned, or 0 otherwise. Looking back to the previous example: \"How can
individuals request corrections for factually inaccurate information about
themselves in ChatGPT output\", we get the following results:

    Total MRR =  1.0 /  1
    # Hits =  1.0 /  1
    Expected ID =  ['f712129a-a58d-4e36-b62f-22ebfeda56a8']
    Retrieved IDs =  ['f712129a-a58d-4e36-b62f-22ebfeda56a8',
                      '1f95b363-1002-4f2d-bf17-ab4842714072']

We can see that the expected document ID was the first one in the
retrieval list, and thus MMR and \# hits are both 1/1. Looking now to a
sample of 50 validation Q&A pairs:

``` {caption="\"Context relevance scores\""}
- Base model:
  - Total MRR =  36.5 / 50
  - # Hits =  42.0 / 50
- Fine-tuned model:
  - Total MRR =  40.0 / 50
  - # Hits =  46.0 / 50
```

We see that the source document was returned in the majority of cases
and was frequently (though not always) the top returned document, but
the fine-tuned model did somewhat better.

Looking now to answer relevance, we can ask whether the RAG pipeline
produces a reasonable answer to our queries. Here, we can use the
RelevancyEvaluator from LlamaIndex:

``` {.python language="Python" caption="\"Answer relevance function\""}
from llama_index.core.evaluation import RelevancyEvaluator

def run_answer_relevance_eval(index, queries):
    query_engine = index.as_query_engine(llm=llm_gpt4)
    ans_relev_evaluator = RelevancyEvaluator(llm=llm_gpt4)
    
    answer_eval_results = []
    for query in queries:
        response = query_engine.query(query)
        ans_relev_eval = ans_relev_evaluator.evaluate_response(
            query=query, response=response)
        answer_eval_results.append(ans_relev_eval)

    return answer_eval_results
```

This function takes in a query, generates a response, and then asks
GPT-4 if the query is responsive to the question. In this case, we get a
simple True or False response. Here again is our test case:

``` {caption="\"Answer relevance example\""}
query = "How can individuals request corrections for factually inaccurate
         information about themselves in ChatGPT output?"
results = run_answer_relevance_eval(index, [query,])

Response:
Individuals can request corrections for factually inaccurate information about
themselves in ChatGPT output by submitting a correction request through
privacy.openai.com or by sending an email to dsar@openai.com. If the inaccuracy
cannot be corrected due to the technical complexity of the models, they can
request the removal of their Personal Information from ChatGPT’s output by
filling out a specific form.

Relevant:
True
```

Evaluating 50 samples from the validation set, we get:

``` {caption="\"Answer relevance scores\""}
- Base model: 47 / 50
- Fine-tuned model: 49 / 50
```

Once again, we see a slight improvement from fine-tuning, this time in
arguably the most important metric: responsiveness of the query to the
question.

The final evaluation metric is answer faithfulness, or "groundedness",
where we ensure that the generated responses are grounded in the
context. For our models, the transformation from context to response is
done by GPT-4 instead of our vector index, so we would probably expect
very good performance and little difference between the two models. For
LlamaIndex, groundedness corresponds to FaithfulnessEvaluator:

``` {.python language="Python" caption="\"Groundedness function\""}
from llama_index.core.evaluation import FaithfulnessEvaluator

def run_groundedness_eval(index, queries):
    query_engine = index.as_query_engine(llm=llm_gpt4)
    faithfulness_eval = FaithfulnessEvaluator(llm=llm_gpt4)

    ground_eval_results = []
    for query in queries:
        response = query_engine.query(query)
        ground_eval = faithfulness_eval.evaluate_response(response=response)
        ground_eval_results.append(ground_eval)

    print(sum([aer.passing for aer in ground_eval_results]))
    return ground_eval_results
```

For our test example about correcting factually inaccurate information,
the response is roughly equivalent to the first piece of context, which
means it passes the Groundedness test. Looking at the set of 50 examples
with both models:

``` {caption="\"Groundedness scores\""}
- Base model: 48 / 50
- Fine-tuned model: 49 / 50
```

As expected, both models perform well, with only a minor difference. It
is interesting to consider the few cases where this did not work. To
take one example, which failed on both models:

``` {caption="\"Groundedness failure example\""}
Query: What actions could potentially prevent OpenAI from being considered a
"service provider" under the CCPA or a "processor" under U.S. Privacy Laws?

Context: Privacy Laws or a “share” under the CCPA (or equivalent concepts under
U.S. Privacy Laws); or (ii) render OpenAI not a “service provider” under the
CCPA or “processor” under U.S. Privacy Laws.

Response: The context does not provide specific actions that could potentially
prevent OpenAI from being considered a "service provider" under the CCPA or a
"processor" under U.S. Privacy Laws.
```

The response here is actually accurate, the context does not explain
what actions create those conditions -- it looks like that information
was probably in the text segment preceding the retrieved context.
However this was flagged as a failure. Interesting, we find an
additional three samples of the 50 that were marked as successful with a
similar non-response answer for the base model (2 in the case of the
fine-tuned model).

<center>

|  **Model**      | **Context Relevance (MMR)**   | **Context Relevance (\# Hits)**         | **Answer Relevance**  | **Groundedness**|
|  -------------- |-----------------------  |--------- |---------------------- |------------------|
|  Base           |         36.5            | 42.0     |         47            |       48|
|  FT             |         40.0            | 46.0     |         49            |       49|
|  Base RR        |         37.5            | 49.0     |         43            |       47|
|  FT RR          |         40.7            | 49.0     |         47            |       48|

*Table 1: Summary of evaluation results (out of 50) on the TruLens triad of
  RAG evaluations for four model setups: base, fine-tuned, base +
  reranking, fine-tuned + reranking.*

</center>

A summary of our results is given in Table 1, along
with two additional model configurations -- the base and fine-tuned
versions combined with reranking (return top 20 \> reranked top 2).
Reranking significantly boosts context relevance, increasing the number
of captured hits to nearly 100% while marginally improving total MRR
score. However reranking has actually decreased the metrics for answer
relevance and groundedness. Why is unclear, but suggests that care must
be taken when incorporating reranking modules -- their utility must be
validated and not just taken as granted.

## Conclusion

In this exercise, we discussed the basic steps of setting up a minimally
functional RAG application. We then test more advanced methods to
improve on the results, and demonstrated how to appropriately evaluate
the responses to ensure that the application works as intended. It is
clear that tools such as LlamaIndex are extraordinarily powerful in
their ability to enrich the knowledge of LLMs without requiring a great
deal of effort or model training.

