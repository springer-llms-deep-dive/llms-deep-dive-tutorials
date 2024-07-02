# Tutorial: Fine-tuning LLMs in a Resource-Constrained Setting

## Overview

We have covered several parameter-efficient fine-tuning techniques and
outlined two major approaches to fine-tuning LLMs: instruction and
alignment tuning. This tutorial will leverage LoRA and QLoRA to train
LLMs to accomplish a specific instruction-based task. While this is not
strictly instruction tuning, as we focus on a single task instead of a
wide range of tasks, our templating approach follows the methodology of
instruction tuning. Note that the results captured here are based on the
performance of a Google Colab session with a 16GB V100 GPU.

**Goals:**

-   Demonstrate the advantages of parameter-efficient fine-tuning in
    terms of both memory requirements and resulting output quality.

-   Examine the relative capabilities of a larger LLM and a scaled-down
    LLM.

-   Implement an evaluation rubric for generated text outputs, using a
    more sophisticated LLM as the grader.

### Tools and libraries

-   **HuggingFace**: Provides the pre-trained models and data needed for
    the experiments and a concise training loop.

-   **PEFT**: A Parameter-efficient fine-tuning (PEFT) library for
    Python, which hosts implementations of LoRA and QLoRA. The latter
    also requires the BitsAndBytes python package, which assists with
    8-bit quantization and functionality.

-   **DistilGPT2**: An adaptation of the GPT-2 autoregressive LLM,
    DistilGPT-2 is an 82 million parameter model trained with the
    supervision of the larger GPT-2.

-   **Llama-2-7B**: A seven billion parameter generative text LLM built
    and released open-source by Meta.

### Datasets

-   **TWEETSUMM**: A dataset of conversations between customers and
    customer service agents over Twitter. The conversations are
    generally  10 messages long and are paired with hand-written
    summaries of the customer's request and the agent's response.
    (Feigenblat et al. 2021)

## The TWEETSUMM dataset

In this tutorial, we create an LLM that can take in a conversation
between a customer and a service agent and return a summary of the
salient points. To do this, we are using the TWEETSUMM dataset. This dataset
consists of back-and-forth conversations between customers and service agents
from various companies on x.com (formerly Twitter). Paired with each
conversation are hand-written two-sentence summaries of the
conversation, noting the customer's request and the agent's response. In
most cases, there are multiple summaries written by different
annotators.

We load these data and apply a small amount of pre-processing, removing
unnecessary links and standardizing the names to \"Customer\" and
\"Agent\" to create back-and-forths paired with a single two-sentence
summary. Here is an example of what this dataset looks like:

``` 
-- -- Conversation 1 -- --
Customer: hi, my Acc was linked to an old number. Now I’m asked to verify
my Acc, where a code / call wil be sent to my old number. Any way that I
can link my Acc to my current number? Pls help 
Agent: Hi there, we are here to help. We will have a specialist contact you
about changing your phone number. Thank you. 
Customer: Thanks. Hope to get in touch soon 
Agent: That is no problem. Please let us know if you have any further
questions in the meantime. 
Customer: Hi sorry, is it for my account: __email__ 
Agent: Can you please delete this post as it does have personal info in it.
We have updated your Case Manager 
Agent: who will be following up with you shortly. Feel free to DM us anytime 
with any other questions or concerns 2/2 
Customer: Thank you 
Agent: That is no problem. Please do not hesitate to contact us with any
further questions. Thank you. 

-- -- Summary -- --
Customer is asking about the ACC to link to the current number. Agent says
that they have updated their case manager.
```

Each conversation has 2-3 hand-written summaries, so we create separate
query/response pairs with each for the training and validation splits to
offer variety to the model. The number of entries in our data splits is:

-   **Train**: 2629

-   **Validation**: 356

-   **Test**: 110

We further reduce the test set to 50 entries for testing in this
tutorial. To convert these data into instruction-based queries, we use
the following template:

``` 
### Instruction:
Read the following conversation between a customer and a customer service
agent, and then create a two sentence summary of the conversation,
describing the customer's question and the agent's response.

### Conversation: 
<tweet-conversation>

### Summary:
<summary> <END_OF_SECOND_SENTENCE>
```

Here `<tweet-conversation>` is the back-and-forth between customer and
agent, `<summary>` is one of the hand-written summaries for that
conversation, and \<END_OF_SECOND_SENTENCE\> is a token we add to assist
the model in knowing when to stop producing summary output. We find this
stopping token helps in producing two-sentence summaries, as the model
tends to keep generating text without it.

>**Grading with GPT-4**

> To assess the quality of the computer-generated
summaries, we establish three criteria that define a summary score.
> 1.  Is the description of the customer's question/complaint reasonably
    accurate?
> 2.  Is the description of the agent's response reasonably accurate?
> 3.  Is the summary two sentences in length?

> The summary receives one point for meeting each of these criteria.
Following Dettmers et al. (2023), we will use GPT-4 to grade the summaries
and assign scores. We pass GPT-4 a rubric with these scoring criteria,
along with the input conversation and generated summary and ask it to
return a score out of 3.

## Fine-tuning DistilGPT-2

We can take our first crack at a chat summary bot with our data in
order. We first test DistilGPT-2, an 85 million parameter autoregressive
LLM trained with supervision from GPT-2, selected because its relatively
low memory requirements allow us to easily fine-tune it on our Google
Colab set up.

As a baseline, we first ask DistilGPT-2 to generate summaries for each
test set conversation without fine-tuning. We define a `transformers`
pipeline for text generation and then pass in prompts from the
templatized TWEETSUMM test set:

    import torch
    from transformers import pipeline

    generator_base = pipeline("text-generation",
                              model='distilgpt2',
                              device='cuda:0',
                              max_new_tokens=100)

    base_responses = []
    for theprompt in tweetsum_datasets['test']['question']:
        with torch.autocast("cuda"):
            base_output = generator_base(theprompt)[0]
            base_responses.append(base_output['generated_text'])

The calls to `generator_base` create the summaries for each templatized
conversation. Here is one illustrative example of the output, given in
response to `Conversation 1`:

``` 
### Summary: 
Please go to our website link below for the full conversation. http://www.briandreysolutions.com/briandreysolutions/briandreysolutions/briandrey-
solutions/briandreysolutions/briand...
```

Unsurprisingly, the output is poor. DistilGPT-2 is too small of an LLM
for any sort of impressive emergent capabilities without additional
fine-tuning, and it ends up repeating endlessly in a made-up URL. Now,
let's fine-tune the model with the training set. For training, we use
the python package `trl`, which implements a convenient wrapper around
the `transformers` functionality.

    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=5
    )

    sft_trainer = SFTTrainer(
        model='distilgpt2',
        train_dataset=tweetsum_datasets['train'],
        eval_dataset=tweetsum_datasets['valid'],
        args=training_args
    )

    sft_trainer.train()

We have not attempted to optimize this model's hyperparameters, choosing
standard values instead. At five epochs of 2629 examples each, the
training loop takes about 10 minutes. Once tuned, the model does a
better job of complying with the instructions, seen here in the
fine-tuned summary of Conversation 1:

```
### Summary:  Customer is asking for the help where a code is attached to an
old number. Agent updated that to delete the post as it does have personal 
info in it and informed to DM for further info. <END_OF_SECOND_SENTENCE>
<END_SENTENCE> <END_OF_SECOND_SENTENCE>>"") <END_SENTENCE> 
<END_OF_SECOND_SENTENCE>>> <END_OF_SECOND
```

This clearly worked better than the base model, but problems remain. It
doesn't quite grasp the request from the customer, and though the
summary for the agent is reasonable, the grammar is poor. At the end of
the first two sentences, it repeats variations of the end-of-summary
token given in the instruction template. This is a convenient format as
we can easily detect the start of the tokens and remove them, leaving
just the two sentences. Moreover, this is a clear improvement over the
version of the template with no end-of-summary token, which tends just
to keep expanding the summary until it reaches the limit of 100 tokens.

To test the overall performance, we generated summaries for 50
conversations in the test dataset using both the base and the tuned
models and graded them using GPT4. The cumulative score for the base
model summaries is 2 out of a possible 150. The tuned model performed
considerably better, scoring 67/150. However, this is still far from
ideal, and notably, 43 points come from having two-sentence summaries,
and only 11 and 13 summaries of the customers' and agents' conversation,
respectively, were deemed accurate by GPT-4. Additionally, we find
grammatical problems throughout the summaries. These suggest that the
fine-tuning has taught the model to generate \<END_OF_SECOND_SENTENCE\>
tokens but still lacks the nuanced understanding of English required to
generate summaries.

## LLama-2-7B

Lacking additional data, we can try to improve by moving to a larger
LLM, whose better knowledge of the language could help improve its
ability to parse what is happening in these messages. To do this, we
adopt Llama-2-7B, a 7 billion parameter autoregressive text-generation
LLM released by Meta in 2023. We create a prediction pipeline with
Llama-2-7B following the same structure as done above for Distil-GPT2,
and generate a summary for Conversation 1:

```
### Summary: 
### Instruction:
### Conversation: 
Customer: hi, my Acc was linked to an old number. Now I’m asked to verify my
Acc, where a code / call wil be sent to my old number. Any way that I can
link my Acc to my current number? Pls help 
Agent: Hi there, we are here to help. We will have a specialist contact you 
about changing your phone 
```

It's still not great, it just repeats the conversation and adds in the
instruction label without extra text. However, we find several
conversations that Llama-2-7B summarizes fairly successfully. Here is an
example:

``` 
-- -- Conversation 2 -- --  

### Conversation: 
Customer: Stuck at Staines waiting for a Reading train, why no information or advice being passed on to passengers on the platform? @SW_Help 
Agent: Hi, due to electric supply problems trains are not running to 
Reading.  
Customer: I know the reason, but I don't think it unreasonable to have 
announcements or staff giving updates...
...
Agent: Hi Adrian, we have just been informed by @115793 engineers that the
line is open again and trains can now run between Wokingham and Reading.
Sorry for the confusion.  
 
### Summary: 
The customer asked why there were no announcements or updates on the platform.
The agent responded that due to electric supply problems trains were not running
to Reading. The customer then asked why the 8.42 was going to Wokingham and the
agent responded that no trains were able to run to Reading. The customer then
asked why the train was terminating at Ascot and the agent responded that they
had just been informed by engineers that the line was open again and 
```

This response scored a 2/3 for having accurate customer and agent
descriptions but a length greater than two sentences. Overall, base
Llama-2 scored 25.5/150 on the test set. Only four summaries were two
sentences long, so the additional 21.5 points come from the accuracy of
the summaries -- this is an impressive improvement over DistilGPT2,
which could produce nothing of value until it had been fine-tuned with
the training set. However, the failure modes are pretty bad and look
nothing like what we want. It is clear that Llama-2 has significant
summarization capabilities without tuning, but you have to get lucky --
the response structure is too inconsistent for reliable use.

To improve the outcomes, let's re-purpose the scripts we used on
DistilGPT2 to fine-tune Llama-2 with the training set:

    sft_trainer = SFTTrainer(
        model,
        train_dataset = tweetsum_datasets['train'],
        eval_dataset = tweetsum_datasets['valid'],
        args = training_args,
        dataset_text_field = 'text',
        max_seq_length=1024
    )

    sft_trainer.train()

    -------

    OutOfMemoryError           Traceback (most recent call last)
    <ipython-input-21-085032d91470> in <cell line: 12>()
         10 )
         11 
    ---> 12 sft_trainer = SFTTrainer(
         13     model,
         14     train_dataset = tweetsum_datasets['train'],

    OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00
    MiB. GPU 0 has a total capacty of 15.77 GiB of which 24.12 
    MiB is free. Process 1060724 has 15.75 GiB memory in use...

Whoops, unfortunately, we have run out of memory on our GPU. At seven
billion parameters, the model weights alone on Llama-2-7B consume around
12GB of memory. This would be most of the memory load if we were just
doing inference. But for fine-tuning, you also store gradients and
optimization tensors, roughly quadruple the total required memory. In a
highly optimized case, PyTorch would have to allocate around 64 GB of
memory, well more than the 16 GB on our V100 GPU. By contrast, DistilGPT
comes in at about 84 million parameters, around 1% of the size of
Llama-2-7B, and thus, its total overhead while fine-tuning is well under
1GB in the most optimized state. It appears we will have to find more
memory-efficient approaches for fine-tuning Llama-2.

## Parameter-efficient fine-tuning

As discussed in section Sect. 4.3.2, low-rank adaptors (LoRA) are a popular
and efficient method for reducing the memory requirements of training.
Instead of fine-tuning the entire weight matrix, we only tune two
low-rank matrices, which are then added to the full weights at inference
time, thus significantly reducing the number of parameters whose
gradients are stored in memory during training.

Let's look at LoRA for Distil-GPT2. The primary difference compared to
the training loop for full-parameter tuning for our LoRA implementation is
that we add a LoraConfig object from the `peft` codebase developed by
Huggingface:

    from peft import LoraConfig

    lora_params = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    ...

    sft_trainer = SFTTrainer(
        model = 'distilgpt2',
        train_dataset = tweetsum_datasets['train'],
        eval_dataset = tweetsum_datasets['valid'],
        args = training_args,
        peft_config=lora_params
    )

    sft_trainer.train()

We will also test an even more efficient version, QLoRA, which involves
quantizing the model weights to 4-bits before applying a LoRA approach
to tuning. This is accomplished by instantiating your model with a
BitsAndBytesConfig, which specifies the details of the quantization.

    from transformers import BitsAndBytesConfig

    bnb_params = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        'distilgpt2',
        quantization_config=bnb_params,
        device_map={"":0}
    )
    model.config.use_cache = False

This model is then passed into the `trl` tuning loop along with LoraConfig.
The relativeperformance of LoRA-tuning and QLoRA-tuning for the TWEETSUM
dataset are shown in Table 1.

<center>

|  Model Configuration    |  Summary score (/150) |  Tuning time (m) |
|  ---------------------  | --------------------- | ---------------- |
|  Base DistilGPT2        |  2                    |  0 |
|  Fine-tuned DistilGPT2  |  67                   |  9.7 |
|  LoRA-tuned DistilGPT2  |  58                   |  6.9 |
|  QLoRA-tuned DistilGPT2 |  52                   |  14.3 |

*Table 1: Final score out of 150 for each DistilGPT-2 approach to tuning on the
TWEETSUMM train set and doing casual inference with the test set. Also listed
are tuning times for each model.*

</center>

Our LoRA-tuned model scores 58,
generating two-sentence summaries in 42/50 cases and accurately
describing nine customers and seven agents. Note that this score of 58
has the same number of training epochs as the fine-tuning approach. By
increasing the number of epochs to 20, we found that the test results
scored 65, demonstrating that further model improvements can be achieved
with hyperparameter exploration. QLoRA does marginally worse.

Despite these lower performance, we observe a smaller total GPU workload
during training. Compared to full-parameter fine-tuning, the maximum GPU
RAM occupancy is lower by 228 MB for LoRA tuning and 336 MB for QLoRA
tuning. This is a non-trivial amount, given that DistilGPT-2's weight
matrix is roughly 356 MB.

We can now test these methods on Llama-2-7B. Using the LoRA and QLoRA training
loops, while additionally quantizing to 8-bit for LoRA tuning, we can enter our
training loops without CUDA errors. We tune for
a single epoch, which takes 75 minutes for the LoRA loop and just 21
minutes for the QLoRA loop. The results are shown in Table 2.

<center>

|  Model Configuration    |  Summary score (/150) |  Tuning time (m) |
|  ---------------------  | --------------------- | ---------------- |
|  Base Llama2-7B         |  25.5                 |  0 |
|  Fine-tuned Llama2-7B   |  Failed               |   |
|  LoRA-tuned Llama2-7B   |  131                  |  75.1 |
|  QLoRA-tuned Llama2-7B  |  125                  |  21.3 |

*Table 2: Final score out of 150 for each Llama-2-7B approach to tuning on the
TWEETSUMM train set and doing casual inference with the test set. Also listed
are tuning times for each model.*

</center>

We find a remarkable improvement in
performance, with the LoRA-tuned test-set evaluation scoring 131/150 and
the QLoRA evaluation scoring 125/131. The improved performance can be
seen from the LoRA-tuned summary of Conversation 1, which GPT-4 gave a perfect
3/3:

``` 
### Summary:  Customer is complaining that his account number is linked to an
old number and now he is asked to verify his account number where a code/call
will be sent to his old number. Agent updates that they have updated their case
manager who will be following up with him shortly and asks not to hesitate to
contact them with any further questions. <END_OF_SECOND_SENTENCE>
<END_OF_FIRST_SENTENCE>
```

Fig. 1 summarizes the test-set evaluation
results of every configuration considered in this tutorial. The two
adaptor-tuned Llama-2-7B models dominate the overall score and are the
best for each grading criterion. We see on the right how the fine-tuned
DistilGPT-2 models effectively learned to limit their summaries to two
sentences but were not able to make them accurate enough for the liking
of GPT-4. Base Llama-2-7B produced an equal number of summaries deemed
accurate as the full-parameter fine-tuned DistilGPT-2 but could not
follow the formatting rules without reinforcement. This shows how
smaller LLMs can be tuned to follow specific instructions but ultimately
cannot compete with the semantic capabilities of large LLMs due to their
low information quantity. Among the Llama-2 tuned models, QLoRA slightly
underperforms LoRA but finished tuning in less than 1/3 of the time.
This trade-off is critical for situations with huge training datasets.
Overall, low-rank adaptor tuning took advantage of the large number of
parameters in the Llama-2-7B model, producing a high-quality and
reliable summarization bot.

![Final scores on the TWEETSUMM summarization task for each inference
framework. On the top, we show raw score out of 150, and on the bottom,
we break down the score into the three criteria: successful customer
summary, successful agent summary, and length (is the response 2
sentences long?). Note that full-parameter fine-tuning for Llama-2-7B
did not produce a model due to memory
constraints.](images/peft_tutorial_barchart.png)
*Figure 1: Final scores on the TWEETSUMM summarization task for each inference
framework. On the top, we show raw score out of 150, and on the bottom,
we break down the score into the three criteria: successful customer
summary, successful agent summary, and length (is the response 2
sentences long?). Note that full-parameter fine-tuning for Llama-2-7B
did not produce a model due to memory
constraints.*

## Conclusion

This experiment shows how smaller LLMs can be tuned to follow specific
instructions but ultimately cannot compete with the semantic
capabilities of large LLMs due to their low information capacity. Among
the Llama-2 tuned models, QLoRA slightly underperforms LoRA but finishes
tuning in less than a third of the time. This trade-off is critical for
situations with large training datasets. Overall, low-rank adapter
tuning took advantage of the large number of parameters in the
Llama-2-7B model, producing a high-quality and reliable summarization
bot.
