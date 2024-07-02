# Tutorial: Preparing Experimental Models for Production Deployment

## Overview

In this tutorial, we revisit the experimental models produced in the
Chapter 4 tutorial. However, this time, rather than
focusing on the training process, we look at some of the steps we might
take if we were preparing to deploy one of these models into a
production application. Several of the tools and techniques discussed
throughout this chapter will be applied and demonstrated. However, we
continue to operate entirely within a Colab notebook environment with
the understanding that many readers probably prefer to avoid the cost of
deploying an actual production-grade inference capability.

**Goals:**

-   Take an open-source evaluation tool and an open-source monitoring
    tool for a trial run.

-   Explore the available capabilities in these tools and how they can
    be useful.

-   Observe whether any new characteristics of our models are revealed
    through this process which might impact whether they are fit for

## Experimental Design

This exercise will focus on several key factors that merit consideration
when endeavoring to take LLM capabilities from concept to production. To
set the stage, we assume a scenario in which two candidate models
emerged from our work in the Chapter 4
tutorial. We aim to compare their relative strengths and weaknesses to
determine which best suits the needs of our hypothetical application
while also considering whether any computational bottlenecks can be
addressed to control inference costs. We then consider the longer-term
implications once our selected model is deployed, demonstrating how we
can ensure that it continues to serve its purpose without any unforeseen
consequences.

First, we will look at model evaluation, which is important in fully
vetting any model's behavior before putting it into operation. In
Chapter 4, we evaluated our models by manually prompting
GPT-4 with a grading rubric. Here we take a similar approach but instead
using an open-source tool called TruLens (<https://github.com/truera/trulens>).
It offers an extensible evaluation framework along with a dashboard to compare
metrics across models. There are a variety of similar solutions on the
market, but TruLens has the advantage of being free, whereas many others
do not.

Next, we briefly examine the inference speed of our models. In practice,
we might want to benchmark performance on different GPU architectures,
and consider various optimizations for each before we would have a real
understanding of the cost of running a given model. However, for this
exercise, we will simply look at how our models are operating on our
Colab GPU.

To conclude the tutorial, we construct a scenario in which our model has
been deployed in production for some time. We now want to see whether it
is still behaving as anticipated or whether anything has changed in our
system that may affect the model's performance. To illustrate, we
deliberately manipulate some test data to create a trend of increasingly
long user prompts. For this final portion of the exercise, we use
another free, open-source tool called LangKit [@langkit].

## Results and Analysis

We begin by demonstrating the `trulens_eval` library using a small
portion of the TWEETSUMM test set. TruLens performs evaluation using
feedback functions. There are options to use both built-in and custom
functions to evaluate models. For this exercise, we choose the coherence
and conciseness stock feedback functions. Under the hood, TruLens wraps
other APIs such as OpenAI and LangChain, providing developers with
several options for which provider they wish to use. Metrics such as
conciseness are obtained through the use of prompt templates.

An example of a system prompt template provided for TruLens evaluations:
``` {caption="An example of a system prompt template provided for TruLens evaluations."}
f"""{supported_criteria['conciseness']} Respond only as a number from 0 to 10, where
0 is the least concise and 10 is the most concise."""
```

We observe the mean scores below by applying both our DistilGPT2 and
Llama-2 LoRA models to the test sample. TruLens uses a scoring system
that ranges from 0 to 1 for all metrics. As expected, the larger Llama-2
model performs better across the board. However, we further note that
while the coherence and conciseness scores seem fairly reasonable, the
summary scores are perhaps slightly low - especially for DistilGPT2. We
can recall that these models appeared to perform quite well in our
earlier tutorial. It is likely that part of the reason for this is
simply that we did not invest much time into the design of the prompt
template within the custom evaluation that we wrote for this exercise.
The coherence and conciseness evaluations are built on validated prompt
templates that are held up against a set of test cases by the developers
of TruLens. This example is a good illustration of how difficult
evaluation can be, and why it can be so valuable to leverage tried and
tested solutions.

<center>

|  Model                  |Mean Coherence   |Mean Conciseness   |Mean Summary Quality|
|  ---------------------- |---------------- |------------------ |----------------------|
|  DistilGPT2-finetuned   |0.66             |0.80               |0.29|
|  Llama-2 LoRA           |0.80             |0.83               |0.60|

  *Table 1: Results of evaluating two candidate models with TruLens. Coherence
  and Conciseness are built into the tool, while Summary Quality is a
  custom evaluation that we provide.*
  
</center>

There are distinct advantages to having a standard format for evaluation
that leverages existing prompts where possible rather than building them
all from scratch. First, it can potentially save time when designing the
evaluation methodology. However, defining these types of abstraction
also enables more seamless automation across various aspects of the
LLMOps system. For instance (although we do not simulate this in our
example), TruLens offers the ability to plug into an application such
that user inputs and model outputs are evaluated in flight for real-time
feedback.

We then shift to another freely available LLMOps tool called LangKit.
LangKit is part of a software suite from WhyLabs that offers monitoring
and observability capabilities. An interesting feature we will explore
is the ability to analyze trends in prompts and responses over time. We
simulate this by creating two separate data batches, or profiles, and
comparing them. We break the data into two small sets consisting of
longer inputs and shorter inputs to create variability in the profiles.
Then, we link to the WhyLabs dashboard, where we can explore many useful
metrics in detail.

![A view of the WhyLabs monitoring dashboard, examining selected metrics
to understand how they are impacted by simulated data drift on the
prompts.](images/whylabs-monitoring.png)
*Figure 1: A view of the WhyLabs monitoring dashboard, examining selected metrics
to understand how they are impacted by simulated data drift on the
prompts.* 

Having now applied both TruLens and LangKit to our TWEETSUMM models and
data, a key observation is that there is in fact some overlap in their
capabilities. However, their implementations are quite different, and
each offers certain advantages that the other does not. TruLens is more
focused on evaluations, and LangKit is more oriented toward logging and
monitoring. Depending on the application, it could make sense to use
both, or it could make sense to choose one over the other. These are
only two of the many LLMOps solutions available; however, some research
is often required to identify the most suitable approach.

## Conclusion

Putting LLM applications into production is a significant undertaking
beyond what we can hope to accomplish in a brief exercise such as this
one. We were, however, able to demonstrate some of the tools that exist
to make this process more manageable. There are a vast number of
different considerations that factor into a model's production
readiness, but fortunately, the developers of tools such as TruLens and
LangKit have designed repeatable solutions for many of them. By building
workflows around these tools, an application can progress to a more
mature state in less time.

