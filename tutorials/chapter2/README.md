# Tutorial: Understanding LLMs and Pre-training

## Overview

In this tutorial, we will explore the mechanics of LLM architectures,
emphasizing the differences between masked models and causal models. In
the first section, we'll examine existing pre-trained models to
understand how they produce their outputs. Once we've demonstrated how
LLMs can do what they do, we will run an abbreviated training loop to
provide a glimpse into the training process.

**Goals:**

-   Inspect the inputs and outputs of a LLM, including the tokenizer.

-   Step through code to demonstrate the token prediction mechanisms of
    both masked LLM's and causal LLMs.

-   Illustrate on a small scale how to train a LLM from scratch.

-   Validate that a training loop is working as intended.


## Experimental Design

The eventual result of this tutorial is to see the pre-training process
at work, but we begin by analyzing the elements of LLM architectures. We
first look at the forward pass, which introduces the various components
and how they operate together to fulfill the language modeling
objective. This code is repeated for both BERT and GPT-2 models to
highlight the similarities and differences between masked (encoder only)
and autoregressive (decoder only) models.

Once we have dissected the steps involved in token prediction, it
becomes natural to understand the LLM training cycle as a typical
backpropagation of gradients through the model layers. We assume basic
familiarity with deep learning and do not spend time exploring the
impact of specific hyperparameters or other details of the training
loop. Readers who need a brief refresher may refer to the appendix.

By the end of the exercise, the code will have yielded a toy model that
has memorized a small chunk of Wikipedia data. The notebook we provide
only includes a training loop for GPT-2 and not for a masked model, but
the reader could easily extend this experiment to other LLMs if they so
desire.

<!-- ## Results and Analysis

In our LLM pre-training experiment, the training loss dropped quickly
while the validation loss remained high. This behavior is depicted in
Fig. 1, and we expect it when the model
overfits the training data. It would take far more documents and
training steps for the model to capture enough information to generalize
well to the validation data, which is unsurprising since the number of
viable token sequences in English is enormous.

![The loss curve obtained as GPT-2 learns the contents of a minimal set
of Wikipedia
documents.](images/pretraining_loss.png)
*Figure 1: The loss curve obtained as GPT-2 learns the contents of a minimal set of Wikipedia documents.*

Although the model hasn't been adequately trained to perform well on the
validation data, we can still see that it has learned a lot from the
training data. To verify, we can test on a training example.

``` {.python language="Python" caption="Accessing Dataset Text Example"}
print(raw_datasets["train"][0]["text"])
# Output:
# William Edward Whitehouse (20 May 1859 – 12 January 1935) was an English cellist.

# Career
# He studied for one year with Alfredo Piatti, for whom he deputised (taking his place in concerts when called upon), and was his favourite pupil. He went on to teach at the Royal Academy of Music, Royal College of Music and King's College, Cambridge...
```

Given the first few tokens, we then confirm that our model can complete
this text for us.

``` {.python language="Python" caption="Generating Text with Model"}
text = "William Edward Whitehouse (20 May 1859 – 12 January 1935) was an English cellist.\n\nCareer\nHe studied for one year with"

model_inputs = tokenizer(text, return_tensors='pt')
output_generate = model.generate(**model_inputs, max_new_tokens=5)
sequence = tokenizer.decode(output_generate[0])
print(sequence)
# Output:
# William Edward Whitehouse (20 May 1859 – 12 January 1935) was an English cellist.
# 
# Career
# He studied for one year with Alfredo Piatti,
```

In this case, the model correctly identified Alfredo Piatti, showing it
has memorized this information from repeated exposure to a specific
Wikipedia article. This gives us confidence that our tokenizer and model
are up to learning language patterns from Wikipedia. Of course, this
does not immediately guarantee that the same training approach will
directly translate to a full-sized dataset. Specific parameters, such as
learning rate, may need to be adjusted.


### Tools and libraries

-   **PyTorch**: An open-source machine learning library for Python that
    provides a flexible and efficient platform for building, training,
    and evaluating various deep neural networks. [@NEURIPS2019_9015]

-   **HuggingFace**: Provides the pre-trained models and data needed for
    the experiments and a concise training loop.

-   **BERT**: A widely used encoder-only model that will be examined to
    understand the masked language modeling objective.

-   **GPT-2**: An archetypal auto-regressive model that will be examined
    to understand the causal language modeling objective.

### Datasets

-   **English Wikipedia**: A commonly used data source for language
    modeling since it is a freely available knowledge base covering many
    topics.

 -->

## Understanding Masked Language Models

The first model we will look at is BERT, trained with masked tokens. For
example, the text below masks the word \"box\" from a well-known movie
quote.

```
text = "Life is like a [MASK] of chocolates."
```

We'll now see how BERT can predict the missing word. We can use
HuggingFace to load a copy of the pre-trained model and tokenizer.

``` {.python language="Python" caption="Python code for initializing BERT tokenizer and model"}
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
```

Next, we'll feed our example text into the tokenizer.

``` {.python language="Python" caption="Tokenizing Input and Displaying Encoded Output"}
encoded_input = tokenizer(text, return_tensors='pt')
print('input_ids:', encoded_input['input_ids'])
print('attention_mask:', encoded_input['attention_mask'])
# Output:
#  input_ids: tensor([[ 101, 2166, 2003, 2066, 1037,  103, 1997, 7967, 2015, 1012,  102]])
#  attention_mask: tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```

The `input_ids` represents the tokenized output. Each integer can be
mapped back to the corresponding string.

``` {.python language="Python" caption="Decoding with Tokenizer"}
print(tokenizer.decode([7967]))
# Output:
# chocolate
```

The model will then receive the output of the tokenizer. We can look at
the BERT model to see exactly how it was constructed and what the
outputs will be like.

``` {.python language="Python" caption="Printing Model Configuration"}
print(model)
# Example Output:
# BertForMaskedLM(
#   (bert): BertModel(
#     (embeddings): BertEmbeddings(
#       (word_embeddings): Embedding(30522, 768, padding_idx=0)
#       ...
#     )
#     ...
#   )
#   (cls): BertOnlyMLMHead(
#     (predictions): BertLMPredictionHead(
#       ...
#     )
#   )
# )
```

The model starts with embedding each of the 30,522 possible tokens into
768 dimensions, which at this point is simply a representation of each
token without any additional information about their relationships to
one another in the text. Then, the encoder attention blocks are applied,
updating the embeddings to encode each token's contribution to the chunk
of text and interactions with other tokens. Notably, this includes the
masked tokens as well. The final stage is the language model head, which
returns the embeddings from the masked positions to 30,522 dimensions.
Each index of this final vector corresponds to the probability that the
token in that position would be the correct choice to fill the mask.

``` {.python language="Python" caption="Model Output and Top Predictions"}
model_output = model(**encoded_input)
output = model_output["logits"]
print(output.shape)
# Output:
# torch.Size([1, 11, 30522])

tokens = encoded_input['input_ids'][0].tolist()
masked_index = tokens.index(tokenizer.mask_token_id)
logits = output[0, masked_index, :]
print(logits.shape)
# Output:
# torch.Size([30522])

probs = logits.softmax(dim=-1)
values, predictions = probs.topk(5)
sequence = tokenizer.decode(predictions)
print('Top 5 predictions:', sequence)
print(values)
# Output:
# Top 5 predictions: box bag bowl jar cup
# tensor([0.1764, 0.1688, 0.0419, 0.0336, 0.0262])
```

Printing the top 5 predictions and their respective scores, we see that
BERT accurately chooses \"box\" as the most likely replacement for the
mask token.

## Understanding Causal Language Models

We'll now repeat a similar exercise with GPT-2.

``` {.python language="Python" caption="Initializing GPT2 Model and Tokenizer"}
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
print(model)
# Example Output:
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (wte): Embedding(50257, 768)
#     (wpe): Embedding(1024, 768)
#     ...
#   )
#   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
# )
```

Comparing the architecture to BERT, there are more similarities than
differences. The initial embedding is similar, except GPT-2 uses a
different tokenizer with a more extensive vocabulary. The structure of
the attention blocks is also identical, but GPT-2 does not mask tokens
within the text. Instead, it only masks tokens at the text's end and
predicts them sequentially. This gives it the ability to generate text
of arbitrary length.

We'll start with a few words that can be used as a prompt for GPT-2 to
complete and then pass them through the tokenizer.

``` {.python language="Python" caption="Preparing Model Inputs with GPT2Tokenizer"}
text = "Swimming at the beach is"
model_inputs = tokenizer(text, return_tensors='pt')

print(model_inputs)
# Output:
# {'input_ids': tensor([[10462, 27428, 379, 262, 10481, 318]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
```

Next, we'll apply the model to the tokenized input. The resulting logits
are the same shape as in the previous example with BERT. Notice,
however, that the information needed to predict the next token is found
in the embedding vector associated with the last input token. Hence, we
use -1 as the index to select from the logits, whereas before, we used
the index of the masked token.

Also, in this example, we use `argmax` to select the token with the
highest probability. This is often referred to as a greedy search. The
predicted token will be the same every time when using greedy search. In
some situations, allowing the model to produce more diverse outputs is
preferable. Numerous other search strategies, such as beam search, can
be used to inject varying degrees of randomness into generative models.

``` {.python language="Python" caption="Generating Next Token Prediction with GPT-2"}
output = model(**model_inputs)
next_token_logits = output.logits[:, -1, :]
next_token = torch.argmax(next_token_logits, dim=-1)

print(next_token)
# Output:
# tensor([257])
```

Now that we know the next token, we add it back to the input so the
model can repeat the process and predict the following token. We also
need to extend the attention mask to the same length.

``` {.python language="Python" caption="Updating Model Inputs and Decoding with GPT-2 Tokenizer"}
model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], next_token[:, None]], dim=-1)
model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], torch.tensor([[1]])], dim=-1)

print(tokenizer.decode(model_inputs['input_ids'][0]))
# Output:
# Swimming at the beach is a
```

We can repeat the previous steps and add another token, after which we
have the following output.

``` {.python language="Python" caption="Decoding Model Inputs to Text"}
print(tokenizer.decode(model_inputs['input_ids'][0]))
# Output:
# Swimming at the beach is a great
```

With HuggingFace, there are many nice abstractions that let us avoid
writing out all the code to produce model outputs. We can take a
shortcut and see what our completed sentence looks like.

``` {.python language="Python" caption="Generating Text with GPT-2 and Decoding the Output"}
output = model.generate(**model_inputs)
print(tokenizer.decode(output[0]))
# Output:
# Swimming at the beach is a great way to get a little extra energy.
```

GPT-2 generated a reasonable sentence from the prompt we gave it.
Auto-regressive models like those in the GPT family can produce text of
unspecified lengths, making them very flexible for many applications. As
a result, they have outpaced BERT-like models in terms of popularity.
However, given their bidirectional nature, masked models can sometimes
be superior for capturing specific text properties.

## Training an LLM from scratch

In the final section of this tutorial, we will shift our focus to
training a GPT-2 model from scratch. Training a highly capable LLM is
challenging without much compute power, so we will first demonstrate
this process on a much smaller scale. The code used in this tutorial
could be applied to the entire dataset with minor modifications.

Considering how expensive it is to pre-train an LLM, it's generally a
good idea to convince oneself that the model will work as expected
before committing to an entire training cycle. One common technique is
to overfit the model to a minimal subset of the data. Suppose the model
architecture is correctly implemented to capture the linguistic features
of the data we plan to train on. In that case, we should expect it to be
able to memorize a few examples quickly. Using Wikipedia for training,
we will demonstrate this effect in the following code.

First, we download the data and select a random set of 50 examples each
for training and validation.

``` {.python language="Python" caption="Loading and Preparing Datasets from Wikipedia"}
from datasets import load_dataset, DatasetDict

dataset = load_dataset("wikipedia", "20220301.en")
ds_shuffle = dataset['train'].shuffle()

raw_datasets = DatasetDict({
    "train": ds_shuffle.select(range(50)),
    "valid": ds_shuffle.select(range(50, 100))
})
```

Next, we download the existing GPT-2 tokenizer. Note that at this stage,
it could potentially be helpful to train our tokenizer if we had reason
to believe our data differed significantly from the original GPT-2. This
isn't the case for our data so we can use the existing tokenizer.

``` {.python language="Python" caption="Initializing AutoTokenizer for GPT-2"}
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

The following code block demonstrates how to tokenize each input text.
The parameters used here will break every document into chunks of 128
tokens, as given by `max_length` and `return_overflowing_tokens`. If the
latter parameter were `False`, then the tokenizer would only use the
first chunk of each document and throw the rest away.

``` {.python language="Python" caption="Tokenizing Datasets with Custom Function"}
context_length = 128

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
```

Once the desired tokenization has been determined, we can load a GPT-2
randomly initialized GPT-2 model from HuggingFace with the appropriate
configuration to accommodate the tokenization scheme.

We're now ready to run the training loop. Since this is a test to ensure
the model can learn our tiny Wikipedia sample, it looks slightly
different than a full-scale training loop. Most importantly, we are
going to train over numerous epochs. This would not be typical when
training with a massive dataset since a model usually learns better from
many diverse examples than from repeated examples.

The code shown here is a high-level abstraction of the training loop
provided by HuggingFace. Within the training process, the randomly
initialized parameters in the GPT-2 model are incrementally updated by a
gradient descent mechanism. The loss at each step decreases as the model
predicts tokens more accurately.

``` {.python language="Python" caption="Setting Up and Running the Trainer"}
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="wiki-gpt2",
    num_train_epochs=100
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"]
)

trainer.train()
```

![The loss curve obtained as GPT-2 learns the contents of a minimal set
of Wikipedia
documents.](images/pretraining_loss.png)
*Figure 1: The loss curve obtained as GPT-2 learns the contents of a minimal set of Wikipedia documents.*

In Fig. 1, the training loss dropped rather quickly while the validation loss remained high. This is the behavior we expect when the model overfits the training data. It would take far more documents and training steps for the model to capture enough information to generalize well to the validation data, which is unsurprising since the number of viable token sequences in English is enormous.

Although the model hasn't been adequately trained to perform well on the validation data, we can still see that it has learned a lot from the training data. To verify, we'll test on a training example.

``` {.python language="Python" caption={Accessing Dataset Text Example}
print(raw_datasets["train"][0]["text"])
# Output:
# William Edward Whitehouse (20 May 1859 – 12 January 1935) was an English cellist.

# Career
# He studied for one year with Alfredo Piatti, for whom he deputised (taking his place in concerts when called upon), and was his favourite pupil. He went on to teach at the Royal Academy of Music, Royal College of Music and King's College, Cambridge...
```

Given the first few tokens, we'll then confirm that our model can complete this text for us.

``` {.python language="Python" caption={Generating Text with Model}
text = "William Edward Whitehouse (20 May 1859 – 12 January 1935) was an English cellist.\n\nCareer\nHe studied for one year with"

model_inputs = tokenizer(text, return_tensors='pt')
output_generate = model.generate(**model_inputs, max_new_tokens=5)
sequence = tokenizer.decode(output_generate[0])
print(sequence)
# Output:
# William Edward Whitehouse (20 May 1859 – 12 January 1935) was an English cellist.
# 
# Career
# He studied for one year with Alfredo Piatti,
```

In this case, the model correctly identified Alfredo Piatti, showing it has memorized this information from repeated exposure to a specific Wikipedia article. This gives us confidence that our tokenizer and model are up to learning language patterns from Wikipedia. Of course, this does not immediately guarantee that the same training approach will directly translate to a full-sized dataset. Specific parameters, such as learning rate, may need to be adjusted.


## Conclusion

We have shown how masked and causal language models can predict tokens.
We then demonstrated that these models can internalize information by
repeatedly attempting to predict these tokens and applying subsequent
weight updates to decrease the loss.
