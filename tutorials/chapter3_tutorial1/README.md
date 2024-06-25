# Tutorial: Prompt vs. Pre-Train/Fine-Tune Methods in Text Classification and NER

## Overview

This chapter has introduced the concept of prompt-based learning and
detailed several potential configurations for prompt and answer shape,
but we have not yet demonstrated one of the most significant benefits of
prompt-based approaches over pre-train/fine-tuning approaches: it's
zero- and few-shot performance. This tutorial will show how prompt-based
learning can achieve better results with fewer training examples than
traditional head-based fine-tuning. While prompting may not consistently
outperform fine-tuning with more significant amounts of data and longer
training cycles, prompt learning has a significant advantage in its
efficiency. It allows LLMs to be adapted to new tasks more quickly and
at a lower cost. It is far less dependent on data volume and quality,
and fewer data means less expensive computation.

Our experiment will directly compare the zero-shot and few-shot
capabilities of the pre-train/fine-tune and prompt-based learning
approaches in their application to text classification and named-entity
recognition. We adopt BERT as the basis for our fine-tuning exercises
for this test. Using PyTorch, supplemented with OpenPrompt for the
prompt-based portion, we will iteratively refine our BERT models with
larger and larger subsets of the training data, predicting on the
validation sets at regular intervals to show how the model is responding
to few-shot learning. Finally, we will compare learning curves for the
two tuning approaches for each NLP task and discuss the implications.

**Goals:**

-   Compare and contrast prompt-based learning with head-based
    fine-tuning.

-   Demonstrate that prompts can be effectively structured to accomplish
    a range of different tasks.

-   Introduce the OpenPrompt library as an example of how the techniques
    discussed throughout the chapter have been implemented.

-   Plot learning curves to illustrate the strong performance of prompts
    in few-shot settings.

## Tools and libraries

-   **PyTorch**: PyTorch is an open-source machine learning library for
    Python that provides a flexible and efficient platform for building,
    training, and evaluating various deep neural networks.

-   **HuggingFace**: Provides the pre-trained model and datasets needed
    for the experiments and a concise training loop.

-   **OpenPrompt**: An open-source tool for prompt-based learning, it
    provides convenient high-level abstractions and handles prompt
    tokenization. We use this package for the prompt-based sections of
    this tutorial.

-   **BERT**: A state-of-the-art masked language model, trained on
    BooksCorpus and English Wikipedia and widely used for fine-tuning
    classification problems. We will fine-tune the `bert-base-cased`
    variant for our experiments. 

## Datasets {#sec:datasets}

-   **SST-2**: A sentiment analysis data set widely used as a text
    classification benchmark [@wang2019glue]. It consists of sentences
    extracted from movie reviews, hand-labeled for positive or negative
    sentiment about the film. The labeled data set is roughly 1800
    samples; from this selection, we take:

    -   512 training examples, half positive and half negative.

    -   872 test examples, comprising the full labeled validation set.

    We use this subset for our text classification experiments. When
    iteratively training, we ensure equal numbers of positive and
    negative samples are in the training set. This avoids the situation
    where a random selection of a small number of data points includes
    many more samples of one label, which can skew results unexpectedly.

-   **CoNLL-2003**: A NER dataset consisting mainly of news headlines
    and quotes, with entity tags for people, organizations, and
    locations, as well as a tag for other miscellaneous named entities
    [@tjong-kim-sang-de-meulder-2003-introduction]. The full labeled
    data set is just over 20,000 samples; from this selection, we take:

    -   1024 training examples.

    -   500 test examples.

    We use this subset for our named entity recognition experiments.
    Note that in NER, each sample consists of many labeled tokens, as
    each word in the sample has an associated entity tag. Consequently,
    these 1024 training sentences consist of some 14,000 labeled tokens.

In both cases, we download and manipulate the samples using the
`datasets` package from *HuggingFace*. This provides a convenient and
easily repeatable procedure for acquiring and using training and
validation data.

## Head-Based Fine-tuning for Classification

We begin with traditional head-based fine-tuning for text
classification. As described in Sect. 3.1.2, this process involves tuning a
task-specific head with sentence/label pairs to enable transfer learning
from the rich language representation of the LLM (in this case, BERT) to
the classification task. We first collect the *SST-2 GLUE*
classification dataset described above and divide it into positive and
negative halves, and tokenize the samples with BERT:

    # Load GLUE dataset
    dataset = load_dataset("glue", "sst2")
    dataset["pos_train"] = dataset["train"].filter(lambda x: x["label"]==1)
    dataset["neg_train"] = dataset["train"].filter(lambda x: x["label"]==0)

    # Tokenize data
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets.set_format("torch")

Each sample in the tokenized data set now consists of three arrays
necessary for the upcoming training loop.

-   **input_ids**, consisting of a list of integers that map each token
    in the sentence to the BERT index associated with those tokens.

-   **token_type_ids**, an array indicating whether each token is in the
    \"question\" or \"answer\" portion of the transformed sentence (in
    our case, the whole sentence questions, hence every entry is 0)

-   **attention_mask**, an array of 1s and 0s, indicating the location
    of tokens to be considered while training (1s) and the location of
    tokens to be ignored (0).

We also define an accuracy metric, which simply calculates the fraction
of predictions that are correct. This metric is standard for binary
classificaiton tasks with roughly equal balance in the train and
validaiton sets.

    def compute_acc(eval_preds):
        # Find highest confidence label
        preds = np.argmax(eval_preds.predictions, axis=-1)
        labels = eval_preds.label_ids
        # Calculate percentage of correct predictions
        acc = sum([int(i==j) for i,j in zip(preds, labels)])/len(labels)
        return acc

For training, we use a highly abstracted and minimalist training loop
implemented with HuggingFace. While there are many ways to further
optimize PyTorch training, this basic structure is simple enough to be
intuitive, while still performing at a level for an effective comparison
against the prompt model. The loop fine-tunes the model with the number
of samples given in `training_sizes`, comprising of zero-shot and number
of few-shot quantities.

    training_sizes = [0, 16, 32, 64, 128, 256]
    for k in training_sizes:
        # Collect training sample of k positives and k negatives
        train_sample = datasets.concatenate_datasets([
            tokenized_datasets["pos_train"].select(range(k)),
            tokenized_datasets["neg_train"].select(range(k))
        ])
        training_args = TrainingArguments("trainer")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=2)
        
        # Define training routine
        trainer = Trainer(model, training_args,
                          train_dataset=train_sample, tokenizer=tokenizer)

        # Do not train for zero-shot (k=0)
        if k > 0: trainer.train()
        # Predict labels for validation sample
        preds = trainer.predict(tokenized_datasets["validation"])
        accuracy = compute_acc(preds)

The results are shown in Table 1:

<center>

|  **\# Train Samples**  | **PT/FT Accuracy** |
|  ------- | ------ |
|  0                     | 0.5092 |
|  16                    | 0.5069 |
|  32                    | 0.6548 |
|  64                    | 0.8486 |
|  128                   | 0.8624 |
|  256                   | 0.8739 |

*Table 1: Prediction accuracy as a function of the number of train samples for
pre-train/fine-tune text classification on the *SST-2 GLUE* data set.*

</center>

For zero-shot, accuracy is almost exactly
50%, no better than random guesses. There is little improvement with the
first few tranches, but a marked improvement by 64 samples, eventually
reaching 87% accuracy with 256. It should be noted that the numbers in
this table are sensitive to precisely which training samples are
selected for the experiment and will vary somewhat in different runs.
However, the basic story they tell does not change.

We have shown that a pre-train/fine-tune approach can achieve high
accuracy after a few hundred samples but has no predictive power in a
zero-shot context. We turn now to a prompt-based training approach to
see how the results compare.

## Prompt-Based Tuning for Classification

In this section, we run
an experiment using the same data in the previous section, but closely
following the prompt-based inferecen and fine-tuning implementation detailed
in Sects. 3.2 and 3.3.5.

We first import BERT for prompting using customized *OpenPrompt* loader
and define a prompt template. This template has a slot for `text_a` and
a slot for the answer, `mask`. Notice how we've framed the desired
output as the 'mask' token and how this follows the underlying
pre-trained language model (PLM) design. Rather than designing a new
task, we give the model a task it already knows how to do.

    from openprompt.plms import load_plm
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")

    template_text = '{"placeholder": "text_a"} it is {"mask"} .'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

*OpenPrompt* supplies a `Verbalizer` object which can be either manual
or automatic (see Sect. 3.4). This part of our model will take
the predicted output and determine which class it best aligns with. In
this case, we have two classes \[0, 1\]. We will assign every example to
0 if 'terrible' is the more likely filled mask or 1 if 'great' is more
likely. Notice how the `label_words` parameter is a list of lists so
that each class can have multiple words.

    verbalizer = ManualVerbalizer(tokenizer, num_classes=2,
      label_words=[["terrible"], ["great"]])

Finally, we adopt the same accuracy metric as in the text classification
experiment. Taken together, these components form the basis of the
prompt-based classification model. As a baseline, observe its
performance as a zero-shot classifier on the validation set.

    prompt_model = PromptForClassification(plm=copy.deepcopy(plm), template=template, verbalizer=verbalizer)
    # Run the zero-shot prediction
    evaluate(prompt_model, val_dataloader)

This prompt model has accuracy in the range of 70% before any training
has taken place. This zero-shot performance is quite impressive compared
to the random predictions produced by the pre-train/fine-tune zero-shot
evaluation; furthermore, rapid improvement can be made by training on
just a few examples.

Next, we iterate over the various training data sizes we defined
previously. This experiment is designed to simulate the effects of
limited available data, so the model is re-initialized to its original
state with a separate training loop for each sample size. We repeat the
training in 5 epochs to ensure convergence.

    for k in training_sizes:
        # Curate training sample and make dataloader
        train_sample = prompt_dataset["pos_train"][:k] + prompt_dataset["neg_train"][:k]
        train_dataloader = PromptDataLoader(train_sample, template,
            tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, 
            shuffle=True, batch_size=4, seed=seed)

        # Load a model instance
        prompt_model = PromptForClassification(plm=copy.deepcopy(plm),
            template=template, verbalizer=verbalizer, freeze_plm=False)

        # Define optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

        # Run the PyTorch loop and evaluate against validation sample
        for epoch in range(5):
            for inputs in train_dataloader:
                logits = prompt_model(inputs)
                labels = inputs["label"]
                loss = loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        evaluate(prompt_model, val_dataloader)

The resulting accuracies are shown in the right-hand column of Table 2:

<center>

|  **\# Train Samples** |  **PT/FT Accuracy** |  **Prompting Accuracy**|
|  --------------------- | --------------------| ------------------------|
|  0                     | 0.5092             |  0.6800|
|  16                    | 0.5069             |  0.6743|
|  32                    | 0.6548             |  0.7867|
|  64                    | 0.8486             |  0.8475|
|  128                   | 0.8624             |  0.8521|
|  256                   | 0.8739             |  0.8658|

*Table 2: A comparison of the prediction Accuracy vs Num Train Samples of
  Pre-train/Fine-tune and Prompt-based text classification for the
  *SST-2 GLUE* data set.*
  
  </center>



With only 32 examples from each class,
the prompt model dramatically jumps in accuracy. It then levels off
quickly and gains relatively little ground with additional data.


Fig. 1 plots the accuracy as a function of training examples for the two
models, starting with zero-shot performance and progressively adding larger
volumes of training data. In contrast to the head-based classifier, the
prompt model achieves impressive results with very few training samples.
The pre-train/fine-tine model eventually becomes competitive with the
prompt-based model but requires 32 training samples (per class) to match
the zero-shot performance of the prompt.

![Graphical representation of the data from Table 2, showing the comparative
accuracy of pre-train/fine tuning and prompt-based learning as a function of
training examples for our text classification exercise.](images/tc_learning_curve.png)
*Figure 1: Graphical representation of the data from Table 2, showing the
comparative accuracy of pre-train/fine tuning and prompt-based learning as
a function of training examples for our text classification exercise.*

Prompt-based learning outperforms the fine-tuning of pre-trained models
at text classification in both the zero-shot and few-shot regimes.
However, their performance becomes commensurate with increasing data
volumes. We will now turn to a second classification task expected to
have poorer zero-shot performance and determine whether prompting still
outperforms fine-tuning in the few-shot context.

## Head-Based Fine-tuning for Named Entity Recognition

NLP's second common labeling task is Named Entity Recognition (NER). In
NER, the goal is to identify the words in a given sentence that belong
to a defined set of entity categories. These categories could include
the names of people, organizations, countries, movies, days of the week,
restaurant dishes; any discrete category in which individual names
distinguish its members. Any word which does not fall into this
description is considered a non-named entity.

Conceptually, NER differs from text classification by solving for
individual labels for each token in a target sentence instead of
assigning a single label to the entire sentence. For the
pre-train/fine-tune approach, this is accomplished by passing a vector
of labels equal in length to the padded, tokenized input at train time
and generating a vector of output labels at inference time. For the
prompting approach, this is accomplished by producing a series of
prompt/answer pairs from each training sentence, where each word is in
the sentence (see Table 3.1 in the book for reference). 

We adopt the *CoNLL-2003* dataset to demonstrate these two approaches,
which provide sentences and associated entity labels for each word.
These data include tags for people, organizations, locations, a
miscellaneous category for other named entities, and a non-entity tag.
Each tokenized sentence in the data set is paired with a vector of equal
length consisting of numbers between 0 and 8. These are the entity tags
mapped by the following dictionary

    label_to_id = {
        0: 'O',
        1: 'B-PER',
        2: 'I-PER',
        3: 'B-ORG',
        4: 'I-ORG',
        5: 'B-LOC',
        6: 'I-LOC',
        7: 'B-MISC',
        8: 'I-MISC' 
    }

The separate B and I tags indicate whether a given NER tag is the first
token for the entity or a subsequent token in a sequence of tokens
comprising a single entity. An example data point:

    dataset = load_dataset("conll2003")

    print(dataset['train'][7158])
    # Output:
    # {'id': '7158',
    #  'sentence': 'Feyenoord midfielder Jean-Paul van Gastel was also named to make his debut in the 18 - man squad.',
    #  'tokens': ['Feyenoord', 'midfielder', 'Jean-Paul', 'van', 'Gastel', 'was', 'also', 'named', 'to', 'make', 'his', 'debut', 'in', 'the', '18-man', 'squad', '.'],
    #  'ner_tags': [3, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

Here \"Jean-Paul van Gastel\" is identified as a person, tagged with
B-PER and I-PER labels, stretching over three tokens, while
\"Feyenoord\" is identified as a one-token organization (B-ORG). The
rest of the tokens are non-entities, demonstrating the preponderance of
these tokens in typical prose text.

For pre-train/fine-tuning, we must first convert the samples to BERT
tokenization, which differs from the default tokenization of the
*CoNLL-2003* data set by including sub-word tokens. This is problematic
for the vector of entity labels as tokens split into sub-words now
require multiple labels for a single word. We adopt the label vector for
each sentence with the following procedure (the processing code may be
found in the accompanying Colab notebook):

1.  Create a new array for the updated entity labels.

2.  For each token in the original vector, determine whether BERT
    tokenization split this word into subvectors.

3.  If no, add the associated tag to the end of the new label array.

4.  If yes, determine the number of sub-words in the token. Append this
    quantity of entity tokens to the new label array, all equal to the
    label's value for the original token.

5.  To account for BERT special tokens at the start and end of the
    sentence, add the label value -100, which is the default ignore
    label for BERT. Add an additional -100s to the end of the vector to
    pad the label vector to the adopted length.

Once this is done, the sentence can be tokenized normally using BERT to
calculate input_ids, token_type_ids, and attention_mask vectors. After
processing, the example shown above now appears like this:

    {
      'id': '7158',
      'tokens': ['[CLS]', 'Fe', '##ye', '##no', '##ord', 'midfielder', 'Jean',
        '-', 'Paul', 'van', 'Gas', '##tel', 'was', 'also', 'named', 'to',
        'make', 'his', 'debut', 'in', 'the', '18', '-', 'man', 'squad', '.',
        '[SEP]', '[PAD]', '[PAD]', '[PAD]', ... ],
      'labels': [-100, 3, 3, 3, 3, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, -100, -100, -100, -100, ... ],
      'input_ids': [101, 11907, 4980, 2728, 6944, 8852, 2893, 118, 1795, 3498,
        12384, 7854, 1108, 1145, 1417, 1106, 1294, 1117, 1963, 1107, 1103,
        1407, 118, 1299, 4322, 119, 102, 0, 0, 0, ... ],
      'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... ],
      'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ... ]
    }

The label vector now has additional entity labels for the sub-words of
\"Feyenoord\" and \"Gastel\". Next, we define a performance metric.
Since most tokens are non-entities we do not wish to detect; we adopt
the F1-score as our metric. This is a natural choice to assess how well
the model identifies true positives while minimizing false positives and
false negatives. Our F1-score routine filters out all tokens that we do
not want to predict on (labeled -100), determines the highest tag
prediction score for each entity, then caluclates precision and recall
to determine F1.

    def compute_ptft_f1(preds, labs):
        # Collect highest prediction score for each entity
        preds_clean = preds.argmax(axis=2)[labs != -100]
        # Collect the labels for each entity
        labs_clean = labs[labs != -100]
        # Determine precision of predictions
        pmask = np.isin(preds_clean, [1, 3, 5, 7])
        p = (preds_clean[pmask] == labs_clean[pmask]).astype(float).sum() / sum(pmask)  # Precision
        # Determine recall of predictions
        rmask = np.isin(labs_clean, [1, 3, 5, 7])
        r = (preds_clean[rmask] == labs_clean[rmask]).astype(float).sum() / sum(rmask)  # Recall
        # Calculate F1-score
        return 2. * p * r / (p + r)  # F1

We generate zero-shot and few-shot predictions using the identical
training loop as in the classification task.
We expand the range of few-shot quantities up to 1024, as we find
empirically that more samples are needed to achieve comparable
performance with the text classification task. The results are shown in
Table 3:

<center>

|  **\# Train Samples**  | **PT/FT F1** |
|  ---------------------- | ------------- |
|  0              |        0.0687|
|  8              |        0.0492|
|  16             |        0.0023|
|  32             |        0.0034|
|  64             |        0.3323|
|  128            |        0.5578|
|  256            |        0.7157|
|  512            |        0.7894|
|  1024           |        0.8526|
                         
*Table 3: F1-scores as a function of the number of train samples of
  pre-train/fine-tune named entity recognition for the *CoNLL-2003* data
  set.*

</center>

Similar to the text classification exercise, without any training, BERT
shows poor performance. This is not surprising, as NER is a label
identification exercise, and BERT does not know what the labels in this
data set signify. It primarily predicts values \> 0, whereas most labels
= 0, thus producing many false positives and a poor F1-score. Looking at
the first few data points, performance worsens as we introduce more
training samples, likely due to catastrophic forgetting. After that,
performance improved with each additional tranche of data, and by 1024
samples, we achieved F1 = 0.85.

## Prompt-based Tuning for Named Entity Recognition

We now consider a prompt-based approach to NER. As with text
classification, we convert the data set into statements and associated
queries. We choose a template form inspired by Cui et al. (2021):


`<sentence>. <candidate_token> is a <entity_type> entity.`


For each sentence in the data set, we create several prompts equal to
the number of tokens with entity tags, where each prompt produced from a
given sentence has the same value of `<sentence>` but a different
`<candidate_token>`. Note that for entities that stretch across multiple
tokens, we create only one prompt where `<candidate_token>` includes
every word in the entity separated by spaces. We provide a fixed list of
entity types defined by the following OpenPrompt verbalizer:

    verbalizer = ManualVerbalizer(tokenizer, num_classes=5,
    label_words=[
      ["non-entity"], ["person"], ["organization"], ["location"], ["miscellaneous"]
    ])

For the sample discussed in the previous section, some of the resulting
prompts look like this:

-   \"Feyenoord midfielder Jean-Paul van Gastel was also named to make
    his debut in the 18 - man squad. Feyenoord is an organization
    entity.\"

-   \"Feyenoord midfielder Jean-Paul van Gastel was also named to make
    his debut in the 18 - man squad. midfielder is a non-entity.\"

-   \"Feyenoord midfielder Jean-Paul van Gastel was also named to make
    his debut in the 18 - man squad. Jean-Paul van Gastel is a person
    entity.\"

-   \"Feyenoord midfielder Jean-Paul van Gastel was also named to make
    his debut in the 18 - man squad. was is a non-entity.\"

-   \"Feyenoord midfielder Jean-Paul van Gastel was also named to make
    his debut in the 18 - man squad. - is a non-entity.\"

We do not have to pre-process the data to BERT tokenization as in the
head-based fine-tuning approach, but we must determine which entities
stretch across multiple tokens (such as Jean-Paul van Gastel). For
prompt-based learning, we have more flexibility in representing entities
and non-entities in the prompt format. The details of this
transformation are in the accompanying Colab notebook. Sample output is
reproduced here:

    print(prompt_dataset['train'][0])
    # Output:
    # {"guid": 0, "label": 3, "meta": {}, "tgt_text": null, 
    #  "text_a": "Sri Lanka and Australia agreed on Friday that relations
    #    between the two teams had healed since the Sri Lankans' acrimonious
    #    tour last year.",
    #  "text_b": "Sri Lanka"}

Comparing this data structure to the template prompt above, `text_a`
corresponds to `<sentence>`, `text_b` corresponds to
`<candidate_token>`, and `label` indicates the `<entity_type>`, which in
this instance maps to \"location\".

We load from OpenPrompt the BERT model and define the template as
before:

    # Load BERT model
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    # Define template
    template_text = '{"placeholder": "text_a"}. {"placeholder": "text_b"} is a {"mask"} entity.'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

Our validation set consists of the same 500 samples tested for the
head-based fine-tune exercise. The performance metric is defined
analogously to the text classification exercise, determining the
F1-score based on the predicted label for each entity from its
associated prompt. We first test zero-shot performance using the same
training and evaluation loop as in the text classification tutorial.

Without any tuning, a prompt-based approach with BERT gives an F1 score
of 7%, approximately the same as the pretrain/fine-tuning approach. In
both cases, we do not do much better than random, in contrast to the
clear zero-shot advantage of the prompting approach in text
classification. This is likely due to the complexity of the NER task
compared to the relatively simple positive/negative sentiment of our
text classification exercise.

Next, we iteratively train the model with larger and larger quantities
of training samples for five epochs and examine the learning curve. The
results are in the right column of Table 4:

<center>

|  **\# Train Samples** |  **PT/FT F1**  | **Prompt F1**|
|  ----------------------| -------------- |---------------|
|  0                    |  0.0687   |      0.0712|
|  8                    |  0.0492   |      0.5788|
|  16                   |  0.0023   |      0.6482|
|  32                   |  0.0034   |      0.7274|
|  64                   |  0.3323   |      0.7867|
|  128                  |  0.5578   |      0.8365|
|  256                  |  0.7157   |      0.8672|
|  512                  |  0.7894   |      0.8304|
|  1024                 |  0.8526   |      0.8551|

*Table 4: A comparison of the F1-scores vs. number of train samples of
  pre-train/fine-tune and prompt-based named entity recognition for the
  *CoNLL-2003* data set.*

</center>


Performance significantly improves with only a few sentences and
gradually increases to 87% F1 at 256 samples. Tuning with larger amounts
of data does not improve performance any further.



As a summary, we compare these results to the pre-train/fine-tune
results in Fig. 2. The comparison is similar to the text
classification situation -- with sufficient data, pre-train/finetune
becomes competitive with prompt-based learning, but in a data-starved
regime, prompt modeling achieves much more impressive results.

![Figure 4: Graphical representation of the data from Table 4,
showing the comparative F1-score performance of pretrain/fine tuning and
prompt-based learning as a function of training examples for our named
entity recognition
exercise.](images/ner_learning_curve.png)
*Figure 4: Graphical representation of the data from Table 4,
showing the comparative F1-score performance of pretrain/fine tuning and
prompt-based learning as a function of training examples for our named
entity recognition
exercise.*

What is particularly striking is that the prompt-based approach does not
undergo a sequence of deteriorating performance for the lowest train
sample quantities as the pre-train/fine-tune approach does. Thus with 32
samples, pre-train/fine-tune gets essentially every single sample wrong,
while prompting has already passed 70% F1. The defining conclusion from
both experiments in this tutorial is that prompt-based learning is the
superior approach when the available training set is limited in
quantity.

## Conclusion

The results of few-shot learning with prompts are very impressive,
especially considering the long-running observation that acquiring an
adequately large set of good-quality training data is the crux of most
machine learning problems. In this prompting paradigm, the key to a
high-quality model is instead the optimal design of prompt templates and
answer formats. A proper selection of these critical ingredients
produces high-quality NLP results with only a few dozen examples. The
following chapter will explore this in greater depth.
