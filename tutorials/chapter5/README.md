# Tutorial: Making a language model more helpful with RLHF

## Overview

This tutorial will demonstrate how reinforcement learning with human
feedback (RLHF) can be used to fine-tune a
generative language model. We use a set of prompts that reflect various
ways a human might interact with a chatbot and a separate reward model
that rates the quality of the generated answers. The reward model
outputs are then used to update the weights of the LM through the PPO
algorithm. The end result is an updated version of the LM optimized to
receive consistently higher returns from the reward model. A schematic of
our plan is shown in Fig. 1.

![An easily accessible demonstration of RLHF using open source models
and data.](images/rlhf-tutorial.png)
*Figure 1: An easily accessible demonstration of RLHF using open source models
and data.*

The methods outlined here reflect key advancements that brought
generative AI into the mainstream and stimulated massive investment.
Before RLHF came into prominence with results such as InstructGPT,
state-of-the-art LLMs could produce realistic prompt answers with
appropriate grammatical usage and accurate factual knowledge. However,
these responses often were not well-suited for addressing a problem or
completing a task in a useful manner. With the addition of RLHF, LLMs
have gained the ability to align their outputs more closely to the
intentions of their users. This has opened the door to many new
applications that require more human-like interactions than chatbots and
virtual assistants were previously capable of. It has also become a
significant contributor to the latest efforts in AI safety.
Bai et al. (2022) did extensive work toward the ideal of \"helpful,
honest, and harmless\" LLM interactions developed through RLHF.

Since RLHF is a costly process in terms of human effort and compute
resources, the experiment provided in this tutorial follows a much
shorter and simpler training process than what would be required to see
awe-inspiring results. But even this small-scale exercise is sufficient
to demonstrate how these techniques have been very effective when
employed at a much larger scale.

**Goals:**

-   Provide a scaled-down view of RLHF, which in practice is an
    expensive and time-consuming endeavor.

-   Examine the components and steps involved in the RLHF process.

-   Test a PPO training loop to see how it improves the responses of a
    selected generative LLM.

### Tools and libraries

-   **PyTorch**: PyTorch is an open-source machine learning library for
    Python that provides a flexible and efficient platform for building,
    training, and evaluating various deep neural networks.

-   **HuggingFace TRL**: This
    is an extension of the HuggingFace Transformers library developed
    for training language models with RL. (<https://huggingface.co/docs/trl/index>)

-   **AI Squared dlite-v1-355m**:
    The model that we will use to demonstrate RLHF training is a GPT
    variant with impressive capabilities for its size. We aim to provide
    an example of RLHF in action with a low barrier to entry. We will
    demonstrate notable improvements in model quality with two to three
    hours of compute time in an easily accessible single-GPU
    environment. Crafting an exercise that meets all of these goals is
    made possible by selecting a model with a much smaller number of
    parameters than the most competitive LLMs currently available.
(<https://medium.com/ai-squared/introducing-dlite-a-lightweight-chatgpt-like-model-based-on-dolly-deaa49402a1f>)

-   **OpenAssistant reward-model-deberta-v3-large-v2**: Since collecting
    human preference data and training a model is a laborious process,
    in this exercise we will take a shortcut by reusing an open source
    reward model that Open Assistant built for their own use of RLHF.
    This model was, in fact, trained on some of the same data we use in
    the tutorial, so it is well-suited to our objective.

### Datasets

-   **Anthropic hh-rlhf**: The dataset used in this tutorial was made
    available by Anthropic along with one of their publications in the
    area of AI alignment. It consists of manually
    written prompts representing human inputs, with two possible answers
    from a virtual assistant. The two answers have been reviewed by
    humans, and in each case one was chosen as being better than the
    other. This dataset is split into a \"harmless\" dataset and a
    \"helpful\" dataset, the latter used in this exercise. From the
    Helpful train set we use:

    -   6,400 training examples

    -   2,354 test examples

    Since the dataset was designed primarily for reward model training,
    each entry contains a conversation that begins with a human asking a
    question followed by an assistant responding. For the purpose of
    this tutorial, we drop the responses and extract the initial human
    inputs as queries for the model to complete during training. Our
    code effectively trains for one epoch, so each query is used
    simultaneously during the RLHF training loop.

## Reinforcement Learning with Human Feedback Experiment

The Anthropic dataset used in this tutorial was developed mainly for the
purpose of training reward models. However, for expediency will not be
training our own reward model in this exercise. Instead, we will extract
the prompts from the text and use them in the RL training loop.
Repurposing this data allows us to sidestep the costly and difficult
initial step of prompt creation. Organizations such as OpenAI generally
have two primary sources for these types of prompts:

-   High-skill contractors who carefully craft prompts and answers,
    which the model can use to learn how to emulate human responses more
    accurately.

-   Online data, meaning that queries submitted by users to an existing
    model are captured. In some cases, those users may also have the
    option of indicating whether or not they like the response, which
    can be even more beneficial to the RLHF fine-tuning process.

``` {#ch5tut:1 .python language="Python" caption="Dataset creation" label="ch5tut:1"}
ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base",
    split="train")

# create prompts by chopping off the Assistant response in the dataset
new_col = [x[len("\n\nHuman: "):x.find("Assistant:")]
           for x in ds["chosen"]]
new_col = [x.replace("\n", " ").strip() for x in new_col]
ds = ds.add_column("instruction", new_col)
ds = ds.filter(lambda x: len(x["instruction"]) < 100)
ds = ds.select(range(6400))
```

The RLHF process begins with an existing pre-trained model. Here, we use
a GPT-like model called DLite, which is relatively small and can be
fine-tuned with limited GPU usage.

The model is loaded twice since the training loop will need a copy to
fine-tune and a static copy for reference.

``` {#ch5tut:2 .python language="Python" caption="Load pre-trained DLite model" label="ch5tut:2"}
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
```

The next step is training a reward model. The reward model aims to score
a given prompt/answer pair according to how favorably a human would
judge the answer. The Anthropic dataset exemplifies this with its
\"chosen\" and \"rejected\" columns. In some cases, rather than only
having two answers to compare, the original LLM is instead asked to
produce several possible answers. Human annotators must then rank them
from best to worst. This data is then used to train the reward model.

There is considerable variation in the choice of reward model
architecture, and the ideal approach remains open for debate. Some have
used reward models similar to the generative model, whereas others have
often used a much smaller reward model, as was the case for InstructGPT.
Intuitively a reward model can be expected to produce better results
with less computation than a generative model, just as a human can
usually rate the quality of an essay with far less thought than it would
take to write an essay. This is a common theme in other realms of
computing as well. Consider for instance, the difficulty of factoring a
large number compared to multiplying the factors and confirming the
result. Many intricacies of how and why RLHF is successful are not fully
understood. Still, the fact that text classification is more
straightforward to optimize than text generation is likely a key aspect.

For this tutorial, we'll avoid spending time training a reward model and
download a popular one from HuggingFace.

``` {#ch5tut:3 .python language="Python" caption="Load reward model" label="ch5tut:3"}
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
rank_model = AutoModelForSequenceClassification.from_pretrained(reward_name)
tokenizer =  AutoTokenizer.from_pretrained(reward_name)
```

At this point, we can begin the RL process. We'll run PPO using the
reward model as the basis for the reward function. The PPO trainer is
initialized with two copies of the generative LLM, in this case, DLite.
One will remain frozen for reference, while the other will be the
initial policy iteratively trained with PPO. The resulting policy after
training is a new version of the DLite model with weights optimized for
better human ratings. In each iteration of the training loop, the
following steps occur:

1.  A prompt batch is passed through the policy LLM and the frozen copy.

2.  The responses from the policy are fed into the reward model to score
    their quality.

3.  Gradient descent is used to update the weights to maximize the
    reward. Note that the reward model is typically not an adequate
    reward function. The construction of the reward function is an
    active area of research. Still, it almost always includes KL
    divergence between the updated policy and the original reference
    LLM. The goal is to ensure that the policy doesn't overfit to the
    reward model and forget too much information that it had previously
    learned.

We arbitrarily set the length of the generated responses to 16 tokens in
our training loop. Based on experimentation, this is long enough for the
reward model to score the output reasonably, but it's still short enough
to keep GPU memory from becoming a significant limitation. Smaller or
larger values could be used; alternatively, one could use some form of
sampling to vary the lengths.

We also set `top_k = 0` and `top_p = 1` for the response generation
step, with `do_sample = True`. These parameters influence whether the LM
will be limited to producing high-confidence responses or if it is
allowed to sample from lower-probability tokens. For the purposes of RL,
we set them such that we will try lower probability tokens as often as
possible to maximize the variation in outputs. This should allow the
algorithm more flexibility to explore unlikely responses rather than
simply exploiting high-probability responses, which will often lead to
local optimality traps.

``` {#ch5tut:4 .python language="Python" caption="Training loop" label="ch5tut:4"}
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.encode(END_KEY)
}

gen_len = 16

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    response_tensors = []
    for query in query_tensors:
        response = ppo_trainer.generate(query, max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    inputs = reward_tokenizer(batch["query"], batch["response"],
        return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    rewards = reward_model.cuda()(**inputs).logits.cpu().detach()
    rewards = [r[0] for r in rewards]

    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    xs.append(stats["ppo/mean_scores"])
```

![With continued training, the policy model becomes increasingly likely
to produce responses that the reward model finds
favorable.](images/rlhf_curve.png)
*Figure 2: With continued training, the policy model becomes increasingly likely
to produce responses that the reward model finds
favorable.*


Both the policy model and the reward model exhibit high variability with
respect to the specific training examples in each batch. Still, the
reward model scores show a visible upward trend, as seen in
Fig. 2 This indicates that the PPO training cycle
is steering the policy toward more preferable responses based on the
reward model's understanding of human values.

## Qualitative assessment

A manual inspection of some examples provides further insight. Below are
several queries with corresponding responses that produced high scores
according to the reward model. The original model that was not trained
with RLHF is given the same prompt for comparison. (Recall that we have
limited the outputs to only 16 tokens for efficiency, so many of these
responses appear incomplete.)

``` {#ch5tut:5 . language="" caption="Comparing responses" label="ch5tut:5"}
query: During which historical or societal events should I absolutely avoid investing money.
response (before): If you want to go on a grand tour of the world and have a ton
response (after): It is important to not invest money into events that have a major impact on international

query: Should you listen to music while studying?
response (before): Yes, it's important to listen to music during study and in conversation to keep
response (after): Preferably without headphones or distractions, but in certain situations, it is possible

query: Can you recommend a good recipe for a vegetarian curry?
response (before): A great plant-based recipe for a vegetarian curry is South Indian Veggie Curry
response (after): A good recipe for a vegetarian curry would be a recipe like " Sweet Lamb K

query: How do I make baked chicken?
response (before): The easiest way to make baked chicken is to put the chicken in a baking dish
response (after): To make baked chicken, preheat the oven to 425°F (190°

query: Do Christmas lights really last longer if you keep them on all the time?
response (before): ### Instruction:No, not really.### Instruction: According to research, lighting the Christmas tree### Instruction:
response (after): No, Christmas lights can't last longer if you keep them all the time.
```

A significant degree of subjectivity is involved in deciding which
response is better. Still, the RLHF model trained with PPO has some
notable advantages over its counterpart. For one, the tuned model
appears to be a little more specific, even suggesting an exact
temperature to bake chicken (in both Fahrenheit and Celsius, no less,
although the conversion is slightly off). It also appears to stay closer
to the topic while at the same time having less of a tendency to repeat
back the words in the original prompt. Not all of the information is
perfectly accurate, such as lamb being a vegetarian recipe, but there's
a noticeable intent to provide answers to each question. On the other
hand, the original model offers a plausible continuation of each
conversation but doesn't always stay entirely on topic.

## Quantitative assessment

To quantitatively measure the gains achieved by the RLHF process, we'll
calculate the perplexity metric widely used for autoregressive models.
It essentially works by referencing a desirable text sample and
computing probabilities that the LM could have produced that exact
sequence of tokens as its output. The negative log-likelihood is
exponentiated; thus, the minimum value is 1.0, with lower scores
representing a better model than higher scores. An excellent visual
explanation of perplexity can be found here:
https://huggingface.co/docs/transformers/perplexity. The HuggingFace
'evaluate' library also provides a convenient method for computing
perplexity, which we use in this tutorial.

We use the `test` split of the Anthropic dataset that the RL policy was
trained on for our evaluation data. This will give us similar types of
prompts, but they are examples that neither model has seen yet. The text
from the `chosen` column is supplied to both the original LM and the one
that we tuned on the Helpful data. This allows us to compare how well
each LM is conditioned to produce an output that a human annotator
considers helpful. We use the following code snippet to
calculate perplexity for both the original and RLHF models:

``` {#ch5tut:6 .python language="Python" caption="Calculating perplexity" label="ch5tut:6"}
from evaluate import load

test_ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")

perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=test_ds["chosen"],
                             model_id=config.model_name)
print(results["mean_perplexity"])
```

Our results show an improvement of more than 20% on the Helpful test
data, confirming that our short RL training loop had the intended effect
of aligning the model's responses to human preferences.

<center>

 | LLM                  |  Helpfulness Perplexity (Lower is better)|
 | ----------------------| ------------------------------------------|
 | Original DLite model  | 31.351|
 | RLHF-tuned model      | 25.680|

*Table 1: Inference perplexity measured using the Helpful test set for the original DLite model and the RLHF-tuned DLite model. The lower perplexity of the tuned model demonstrates improvement in human-like response quality.*
 
</center>
 
 
## Conclusion

The results in this tutorial illustrate how RLHF can be an effective
technique for aligning language models to desirable human values and
intentions. This process is typically far more costly, involving larger
models and longer training cycles. The advancements, however, have been
well worth the price of admission for companies successfully utilizing
RLHF. It played a critical role in the recent breakthrough in chatbot
capabilities and continues to be an essential area of research
concerning AI safety.

