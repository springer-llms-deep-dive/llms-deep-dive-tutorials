# Tutorial: Approaches to prompt engineering

## Overview

One of the central themes of Chapter 3 is the use of template engineering
to improve the analytic capabilities of prompt-tuned LLMs. In Sect. 3.2.2, 
we demonstrated the sensitivity of
LLM inference outputs to choices in template architecture and the fine
details of prompt composition. That demo was accomplished with a web
application, a useful proof-of-concept but inherently limited in its
capabilities as it cannot be fine-tuned. Therefore in this tutorial, we
will expand on these exercises by exploring few- and many-shot
prompt-tuning, discussing results for variable prompt template designs,
and aiming to grasp the critical importance of prompt template
optimization.

**Goals:**

-   Illustrate that task performance is highly sensitive to prompt
    template design, with even subtle variations making a notable
    difference.

-   Explore some of the factors that lead to higher quality prompt
    templates.

-   Conduct automatic tuning with soft prompts to demonstrate how they
    compare to manually constructed prompts.

## Experimental Design

### Tools and libraries

-   **PyTorch**: PyTorch is an open-source machine learning library for
    Python that provides a flexible and efficient platform for building,
    training, and evaluating various deep neural networks.

-   **HuggingFace**: Provides the pre-trained model and datasets needed
    for the experiments and a concise training loop.

-   **OpenPrompt**: An open-source tool for prompt-based learning, it
    provides convenient high-level abstractions and handles prompt
    tokenization. We use this package for the prompt-based sections of
    this tutorial (Ding et al. 2021)

-   **T5**: A large language model built using the encoder-decoder
    paradigm and trained on hundreds of gigabytes of cleaned internal
    crawl data. (Raffel et al. 2020)

### Dataset

-   **SuperGLUE BoolQ**: A natural language inference and boolean
    question/answer dataset (Clark et al. 2019). Each `BoolQ` entry
    consists of a contextualizing paragraph, a yes or no question
    related to, and answered within the contents of the paragraph, and a
    positive or negative label denoting the answer to the question. The
    full dataset is nearly 16,000 entries long. For our training and
    testing, we collect three splits:

    -   5000 training examples.

    -   1000 samples for a validation set, half positives, and half
        negatives.

    -   1000 samples for a test set, half positives, and half negatives.

    We use this subset for our experiments on natural language inference
    through prompt tuning. When iteratively training, we ensure equal
    numbers of positive and negative samples are in the training set.
    This avoids the situation where a random selection of a small number
    of data points includes many more samples of one label, which can
    skew results unexpectedly.

## Manual Template Engineering

This tutorial will consider several different approaches to template
engineering and assess their performance in training a model against a
benchmark dataset. We begin with the simplest approach: manual template
engineering (see Sect. 3.3.2). In manual template engineering, the
user's task is to create a template that best suits the task. One can
reference the existing literature suggesting templates for all
prompt-based learning tasks (see e.g. Table 3.1 in the book), or
experiment with different configurations.

In this tutorial, we consider the *SuperGLUE BoolQ* data set, which
provides triplets of an informational paragraph, a yes or no question
related to the paragraph's content, and the correct response. Here is an
example datum from this set:

``` {language="" caption="GLUE BoolQ example" label="code:ch4tut_1"}
passage: "Look What You Made Me Do" is a song recorded by American singer-
songwriter Taylor Swift, released on August 24, 2017 by Big Machine Records
as the lead single from her sixth studio album Reputation (2017). Swift wrote
the song with her producer Jack Antonoff. "Look What You Made Me Do" is an
electroclash and pop song, with lyrics about various issues that built Swift's
reputation. Right Said Fred band members Fred Fairbrass, Richard Fairbrass, and
Rob Manzoli are also credited as songwriters, as it interpolates the melody of
their song "I'm Too Sexy" (1991).

question: "did taylor swift write look what you made me do"

label: 1
```

The BoolQ dataset is very expansive in its topics coverage, including
history, science, geography, law, sports, pop culture, and more, making
it a fascinating dataset for exploring LLMs' natural language inference
capabilities. These three components will be incorporated into a prompt
template and the LLM tuned with these samples.

To illustrate the importance of manual template choice, let's start with
the simplest conceivable prompt: just the three components.

``` {#code:ch4tut_2 . language="" caption="Simplest manual prompt" label="code:ch4tut_2"}
text = "{passage} {question} {mask}"

mask_verbalizer = {0: 'no, 1: 'yes'}
```

where `{passage}` is the context paragraph, `{question}` is the associated
query, and `{mask}` is the answer to `{question}`, verbalized with 'yes' and
'no' corresponding to 1 and 0, respectively. Leveraging *OpenPrompt*, we
implement this example as follows:

``` {#code:ch4tut_3 .python language="Python" caption="Simplest manual prompt implementation" label="code:ch4tut_3"}
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification

plm, tokenizer, model_config, WrapperClass = load_plm('t5', 't5-base')
base_temp = '{"placeholder":"passage"} {"placeholder":"question"} {"mask"}'
verbalizer = ManualVerbalizer(tokenizer, num_classes=2, 
                              label_words=[['no','No'],['yes','Yes']])

template = ManualTemplate(tokenizer=tokenizer, text=base_temp,
                          verbalizer=verbalizer)

prompt_model = PromptForClassification(plm=plm, template=template,
                                       verbalizer=verbalizer)
```

Using a similar *PyTorch* training and evaluation loop as used in the first
Chapter 3 tutorial, we run prompt-based tuning with the simple
template on a `t5-base` model for several data sample quantities. The
results are in the left column of Table 1:

<center>

|  **Num Train Samples** |  **Simplest Template**  | **Simplest + Punct** |  **Standard Template**|
|  --------------- | -------------- | ---------------- | ------------- |
|  0             |          0.501         |          0.526        |          0.564|
|  16            |          0.511         |          0.508        |          0.489|
|  32            |          0.513         |          0.482        |          0.503|
|  64            |          0.518         |          0.516        |          0.535|
|  128           |          0.526         |          0.598        |          0.638|
|  256           |          0.578         |          0.628        |          0.686|
|  512           |          0.546         |          0.638        |          0.640|


*Table 1: A series of zero- and few-shot accuracy scores using *SuperGLUE
  BoolQ* for three different prompt templates: the simplest manual
  prompt, the simplest + punctuation prompt, and the standard manual prompt.*

</center>

Overall, the model performance is poor:

-   Zero-shot inference predicts the negative class for every sample,
    thus reproducing with its accuracy score the ratio of negative to
    total samples in the validation set (50/50).

-   The few-shot examples do better, but only marginally better than
    random -- not far from flipping a coin for each query.

-   Model performance peaks with around 256 samples but never achieves
    impressive results.

This outcome is unsurprising, given the imprecision and ambiguity of the
prompt template. Note that we did not individually tune the model
hyperparameters in this experiment; instead, we adopted a uniform set
across each template.

The simplest way to improve this template would be to enforce
punctuation. Many samples do not end with periods, and the questions do
not end with question marks, so the samples run together in the prompt.
So we test the simple change of adding a period to the passage if
missing and a question mark to the end of the question.

``` {#code:ch4tut_4 . language="" caption="Punctuated manual prompt" label="code:ch4tut_4"}
text = "{passage} . {question} ? {mask}"

mask_verbalizer = {0: 'no, 1: 'yes'}
```

The result of this minor change, shown in the middle column of Table 1, is
interesting. The zero-shot performance improves a bit—from ~50% to ~53%—simply
from adding a "`?`" and a "`.`" in the appropriate places.

> **The same\... but different?**

>50% and 53% are not too far apart, but
inspecting the actual predictions reveals an interesting fact: the
simplest template answers \"no\" for nearly every query (99.1%), whereas
the punctuated template answers no on a much more balanced quantity,
34.4%. Thus, although the accuracy is still poor, the punctuated
template induces the LLM to try to answer the question yes or no,
whereas the former does not.

Once prompt tuning begins, the punctuated template improves more rapidly
than the most straightforward template, indicative of improved
prompting. In the few-shot context the punctuation marks help the LLM
parse apart the context paragraph and the question, and identify the
question portion as a query for the model. This shows both how minor
details can have significant performance consequences and how important
it is to instruct the model properly as to what task it should
accomplish. Finally, we test a more standard manual template for this
problem.

``` {#code:ch4tut_5 . language="" caption="Baseline manual prompt" label="code:ch4tut_5"}
text = "hypothesis: {passage} premise: {question} The answer was {mask} ."

mask_verbalizer = {0: 'no, 1: 'yes'}
```

This template should produce better results, as it provides helpful
context around the passage and question portions and precisely queries
the model for an answer. Running the same prompt-tuning experiment described 
above but with the improved template gives the right-hand column in Table 1.

The improvement is notable; its zero-shot performance is the best of the
three templates. However, with a small number of tuning examples,
accuracy declines due to catastrophic forgetting before beginning to
increase again with further tuning. After 256 samples, the model
correctly answered ~69% of prompts, a significant
improvement over the other templates. Further tuning shows a modest
decline in predictive power, which can result from non-optimal
hyperparameter values.

Fig. 1 depicts the three learning curves. They tell a similar story as the
numeric learning curves -- the more explicitly and precisely the
template lays out the task that the LLM is supposed to accomplish, the
better the performance on average. The three X marks on the right-hand
side show each model's performance against a separate holdout set that
we call the test set; we find the exact ordering of template quality
with this independent set.


![Figure 1: The change in SuperGLUE BoolQ validation set accuracy for a model trained with three different prompts. Template shape impacts both zero-shot and few-shot performance in solving the question/answer task.](images/ch4tut_fig1.png)
*Figure 1: The change in SuperGLUE BoolQ validation set accuracy for a model trained with three different prompts. Template shape impacts both zero-shot and few-shot performance in solving the question/answer task.*

It is also clear that this particular task requires a sizable number of
training samples to achieve impressive performance. This contrasts with
the results of the first Chapter 3 tutorial, which showed remarkable gains
with very few samples. We expect this results from the greater
complexity of the *SuperGLUE BoolQ* task compared to named-entity
recognition with *CoNLL-2003*.

This comparison demonstrates the importance of optimizing
templates for the task. However this conclusion suggests an obvious
follow up question: how do we know our template is optimized? In this
section, we adopted a reasonable choice, but it is not guaranteed to be
optimal. In the following sections, we will investigate some template
optimization approaches.

## Manual Template tuning

Manual tuning is the first and most intuitive approach to template
tuning. Here you generate by hand several different variants of the
prompting template, optionally both *cloze-* and *prefix-* style
prompts (see Sect. 3.4.1), and testing their performance. We
will compare the performance of several variants:

<center>

| Example # | Prompt |
| --------- | ------ |
|1 | "hypothesis: {passage} premise: {question} The answer was {mask} ." |
|2 | "{passage} Question: {question} ? The answer was {mask}" |
|3 | "Information: {context} . Question: {question} ? The answer is {mask}" |
|4 | "Context: {passage} Based on the information in this paragraph, {question} ? {mask}" |
|5 | "Question: {question} ? Is the answer yes or no? {mask} . Context: {passage}" |
|6 | "Context: {passage} . Based on this, of the two options (yes or no), {mask} is the answer to this question: {question} ?" |
|7 | "premise: {question} ? The answer was {mask} ."|
|8 | "The answer to the question "{question}" is {mask}"|
|9 | "{question} ? answer: {mask}"|
|10 | "{question} ? {mask}"|

*Table 2: Ten different manual prompt templates for the SuperGLUE BoolQ task.*

</center>

Example 1 repeats the best-performing prompt from the previous section.
The other templates fall into three categories:

-   Examples 2-4 are variants of example 1, mixing up the static words
    between the sample text. Each of these is a *prefix* prompt.

-   Examples 5 and 6 use the *cloze* prompt shape, placing the mask in
    the interior of the sentence.

-   Examples 7 through 10 exclude the contextualizing paragraph and
    treat the prompt more like a information retrieval exercise,
    investigating what knowledge the LLM has encoded (see Sect. 3.2.3).


With each of these templates, we repeat the exercise described above,
fine-tuning the model with prompts and exploring the change in
performance as a function of train samples. The results are shown in the
three panels of Fig. 2. Each panel shows the range of validation set
accuracy scores achieved by the three groups described above: *prefix*
prompts on the left, *cloze* prompts in the middle, and information
retrieval prompts on the right. Each case is benchmarked against the
best-performing template from the previous exercise.

![Figure 2: The range of accuracy scores as a function of training samples for
three different categories of prompts: *prefix*, *cloze* and information
retrieval, described in Table 2. These are compared against the
best-performing prompt from Table 1.](images/ch4tut_fig2.png)
*Figure 2: The range of accuracy scores as a function of training samples for
three different categories of prompts: *prefix*, *cloze* and information
retrieval, described in Table 2. These are compared against the
best-performing prompt from Table 1.*

The results reveal a few interesting features about templates. First,
there is some coherence of behavior within each of the categories.

-   The prefix prompts have some success in zero-shot mode, degrade due
    to catastrophic forgetting with a small number of tuning samples,
    and then improve greatly in predictive power.

-   The cloze prompts do somewhat worse in the zero-shot mode and
    degrade somewhat with a small number of training samples, but after
    that, perform better, eventually reaching parity with the prefix
    prompts.

-   Prompts that provide less context are notably worse. They do a
    little better than random in zero-shot and only do a few percentage
    points better after the full suite of training examples. However, it
    is noteworthy that each prompt does better than random after the
    full train -- the model does encode the answers to some of these
    questions.

-   There is a significant scatter in overall performance within each
    category, which tends to increase with greater training data. This
    suggests that minute differences in template structure can have
    meaningful consequences.

We can reach a few conclusions from this. First, the out-of-the-box `T5`
model possesses minimal information retrieval capabilities for this
dataset and structure. This is not too surprising: the questions in
`SuperGLUE BoolQ` tend to relate directly to the contents of the
paragraph. They are not as general as the style of questions most suited
to information retrieval.

Second, the model is learning to extract information from the context
paragraph. We know this because the *prefix* and *cloze* prompt models,
which contain the context paragraph, perform better after training than
the information retrieval prompts, which do not include the context
paragraph. The LLM's ability to detect and use this information improves
with additional training samples, though not always monotonically.
Evaluation metrics can rise and fall from step to step in our exercise,
suggesting that hyperparameter optimization and fine-tuning of the input
dataset are required to get the most out of prompt-tuned models.

Users can optimize their LLMs by creating and testing many template
configurations. However, the human imagination is an important
limitation of manual template tuning. We are limited to only the
templates that we can think up. In the next section, we will explore
ways to go beyond human imagination and automate template tuning to
refine our prompt shape further.

<center>

|  **\# Samp** |  **#1** |  **#2** |  **#3** |  **#4** |  **#5** |  **#6**  | **#7**  | **#8** |  **#9**  | **#10**|
|  -------------| --------| -------- |--------| -------- |-------- |-------- |-------- |--------| --------| ---------
|  0        |    0.564  |  0.541  |  0.573  |  0.524  |  0.531  |  0.516  |  0.495  |  0.498  |  0.531 |   0.494
|  16       |    0.489  |  0.504  |  0.500  |  0.499  |  0.502  |  0.485  |  0.492  |  0.487  |  0.506 |   0.496
|  32       |    0.503  |  0.531  |  0.551  |  0.531  |  0.519  |  0.491  |  0.457  |  0.504  |  0.512 |   0.507
|  64       |    0.535  |  0.544  |  0.608  |  0.499  |  0.501  |  0.525  |  0.510  |  0.500  |  0.511 |   0.507
|  128      |    0.638  |  0.546  |  0.531  |  0.529  |  0.526  |  0.539  |  0.508  |  0.510  |  0.518 |   0.505
|  256      |    0.686  |  0.596  |  0.704  |  0.616  |  0.560  |  0.686  |  0.522  |  0.540  |  0.544 |   0.537
|  512      |    0.640  |  0.611  |  0.671  |  0.610  |  0.676  |  0.551  |  0.557  |  0.566  |  0.516 |   0.564

  *Table 3: Validation sample accuracy results for each template enumerated in
  Table 2. These data are also represented graphically in Figure 2.
  Template #1 is identical to the standard manual template in the previous
  section. Templates #2-4 are prefix style templates,
  #5-6 are cloze style templates, and #7-10 are information retrieval
  templates, excluding contextualizing paragraphs.*

</center>
                                                                                                 

## Automatic Template tuning

There are several approaches to tuning prompts programmatically using
training data. We have discussed some in Sects. 3.3.3 and 3.3.4.
In contrast to manual prompting, soft
prompting using a variable template which training data can tune. The
soft prompt is initialized with a template that combines the dataset
features with "soft" tokens, which themselves may optionally be
initialized to a given word or phrase, and refines the respective
embeddings through backpropagation to achieve the classification or
generation task of the training data. We implement the soft prompt
approach as follows:

``` {#code:ch4tut_7 .python language="Python" caption="Soft prompt implementation" label="code:ch4tut_7"}
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, ManualVerbalizer
from openprompt import PromptForClassification

plm, tokenizer, model_config, WrapperClass = load_plm('t5', 't5-base')
verbalizer = ManualVerbalizer(tokenizer, label_words=[['no','No'],['yes','Yes']],
num_classes=2)
template = SoftTemplate(tokenizer=tokenizer, verbalizer=verbalizer, text=base_temp)

prompt_model = PromptForClassification(plm=plm, template=template,
                                       verbalizer=verbalizer)
```

Compared to the implementation above, the template object has changed from
`ManualTemplate` to `SoftTemplate`. The variable `base_temp`  in the 
`SoftTemplate` defines a
starting point, which is then iteratively refined. For example, we test
the two following soft template choices:

``` {#code:ch4tut_8 . language="" caption="Soft prompt templates" label="code:ch4tut_8"}
base_temp = "{passage} {question} {mask}"
base_temp = "hypothesis: {passage} premise: {question} The
            answer was {mask} ."
```

We instantiate soft prompts with each of these templates, and in each
case finetune the template with 128 `SuperGLUE BoolQ` samples for
several epochs. For this test, the `t5-base` LLM is frozen, so only the
prompt is tuned. We show these two models' changing validation set
performance in the left panel of Fig. 3.

-   The red dashed line shows the featureless prompt, which fails to
    improve despite 60 epochs of fine-tuning. Given the sparsity of this
    template, the features that could be fine-tuned are simply lacking,
    so no fine-tuning improves the performance.

-   The black line shows the second prompt. Here, we do see significant
    improvement with additional fine-tuning, with the accuracy
    increasing by approximately 4.5% over 60 epochs. The template has
    arrived at a better state than our input template due to soft-prompt
    tuning.

Notably, the performance of the engineered soft-prompt model after 60
epochs out-performs the zero-shot performance of any model shown in Fig. 2,
and indeed any of the models before roughly 128 training samples. This is
achieved with a smaller investment of computing power, given the much
smaller size of the prompt compared to the LLM itself. Thus, for a
situation with a limited number of data points, prompt tuning may be
preferable to LLM tuning when considering the computation expenses of
training.

![Figure 3: *Left:* Results of soft prompt tuning starting with the featureless
prompt (dashed line) and the engineered from defined above (solid line).
*Right:* Learning curves for the four modes of learning described just below.
Our model, which allowed simultaneous prompt and LLM tuning, performed the
best at all stages of the training process.](images/ch4tut_fig3.png)
*Figure 3: *Left:* Results of soft prompt tuning starting with the featureless
prompt (dashed line) and the engineered from defined above (solid line).
*Right:* Learning curves for the four modes of learning described just below.
Our model, which allowed simultaneous prompt and LLM tuning, performed the
best at all stages of the training
process.*

What about more data-rich situations? Does pure soft prompt tuning still
compete with LLM tuning for the `BoolQ` dataset? We explore these
questions by comparing four different tuning approaches.

``` {#code:ch4tut_9 . language="" caption="Soft prompt + LLM tuning options" label="code:ch4tut_9"}
1. Soft prompt tuning only
2. LLM tuning only
3. Soft prompt tuning, then LLM tuning
4. Simultaneous soft prompt and LLM tuning
```

The first and second options represent the approaches in
Fig. 1 and the left panel of Fig. 3. For option 3, we take the prompt
resulting from 60 epochs of refinement shown in black on the left of
Fig. 3, freeze the parameters of the prompt, unfreeze the parameters of the
`t5-base` model, and proceed with model tuning as before. Finally, for
option 4, we begin with an untuned soft prompt, instantiated with the
engineered template from Fig. 3, and an untuned `t5-base` model, unfreeze the
parameters in both portions and fine-tune with the training data.

Each is tuned with 512 samples, 8 epochs, a prompt learning rate of 0.4,
and an LLM learning rate of 5e-4. The results are shown in the right
hand panel of Fig. 3. A few observations:

-   First, for solving the BoolQ dataset, it is clear that LLM tuning is
    advantageous over pure soft prompt tuning. The prompt-only model
    shows significant improvement with additional tuning, but at 512
    samples, it is well below the performance of all three models, which
    allowed the LLM variables to vary. This performance gap should
    narrow with longer training times, as shown by Lester et al. (2021), who
    achieved a performance score around 0.9 with well-optimized training
    and 30,000 training steps.

-   Of the three other models, the one initialized with a tuned soft
    prompt shows the best zero-shot performance, which is unsurprising
    given that it has already been exposed to the training data. After
    that, each model allowing for a nonstatic LLM shows similar
    improvement rates with additional training samples. At each training
    step, there is a small preference for the model where both the LLM
    and prompt are simultaneously tuned over the model, tuning only the
    LLM. This is likely due to the larger number of parameters being
    tuned in the prompt+LLM model (248 million vs. 222 million),
    representing the highest-performing model in this tutorial. However,
    the gain over a well-engineered prompt+LLM tuning is fairly small.

We have shown that soft-prompting (in combination with model tuning) can
outperform manually engineered prompts because of their data-driven
nature and refined embedding states that do not map precisely onto human
text. This second point is key -- manual prompts are generated through
embedding human text. In contrast, soft prompts are refinements from
these initial prompts that can take on quantities with no exact textual
analog. This allows for interpolation between text embeddings, thus
significantly increasing the domain of possible embedded prompts --
similar to how decimal numbers can more precisely describe quantities
than integers. As always, the downside is that more computer time is
required when you increase the number of parameters you are tuning.

## Conclusion

We have shown the vital importance of prompt engineering in optimizing
LLM performance. To be sure, many additional parameters must be
fine-tuned to achieve peak performance that we have not focused on,
including the size of the training set, the number of training epochs,
learning rates, and more. It is also possible to modify the verbalizer
in tandem with the prompt for a slight additional edge in performance.
Nonetheless, from our weakest performing to best performing model, we
have shown an improvement over 25% in prediction accuracy solely from
template engineering.

Thus, great attention must be paid to this component of any prompting
model. Determining which style of prompt engineering is correct for the
problem you are solving is an experimental process that your data will
determine, your machine learning task, your available compute resources,
and your model performance requirements. Testing multiple approaches is
the best path forward to finding the right combination of model and
prompt style to tackle any given machine learning task.
