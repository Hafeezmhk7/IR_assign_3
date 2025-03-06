# Assignment 3: Learning to Rank <a class="anchor" id="toptop"></a>

## Introduction
Welcome to the Learning to Rank (LTR) assignment. This assignment is divided into two chapters. In the first chapter, you will explore offline LTR methods, while in the second part, you will work with learning to rank from interactions.

In Chapter 1, you will learn how to implement methods from the three approaches associated with learning to rank: pointwise, pairwise, and listwise. In Chapter 2, you will simulate document clicks and implement biased and unbiased counterfactual learning to rank (LTR).

**Learning Goals**:
- Extract features from (query, document) pairs
- Implement pointwise, pairwise, and listwise algorithms for learning-to-rank
- Train and evaluate learning-to-rank methods
- Simulate document clicks
- Implement biased and unbiased counterfactual learning to rank (LTR)
- Evaluate and compare the methods

## Scoring and Submission Guidelines for Assignment 3  

To achieve a full score on **Assignment 3**, you must complete both the **implementation** and **analysis** components. The weight distribution is as follows:  

- The **implementation component** accounts for **7/20 of your Assignment 3 grade**. Your score for this component is determined by the number of autograding tests you pass. To maximize your score, ensure that your implementation meets all specified requirements.  
- The **analysis component** constitutes **13/20 of your Assignment 3 grade**. This part will be **manually graded** based on the clarity, depth, and correctness of your explanations and insights. To maximize your score, ensure that your analysis is thorough and well-organized. You will need to have the implementation component finished to complete the analysis.


### Submission Requirements
- You will implement your **code in the provided Python files**.
- Push your code to your **GitHub repository**.
- Write a **concise report in LaTeX** summarizing your findings.  
- Submit your **compiled PDF of the report** via **Canvas** before the deadline.  

## Guidelines

### How to proceed?
We have prepared a notebook: `assignment3.ipynb` including the detailed guidelines of where to start and how to proceed with this part of the assignment

You can find all the code of this assignment inside the `ltr` package. The structure of the `ltr` package is shown below, containing various modules that you need to implement. For the files that require changes, a :pencil2: icon is added after the file name. This icon is followed by the points you will receive if all unit tests related to this file pass.

📦ltr\
 ┣ 📜dataset.py :pencil2: \
 ┣ 📜eval.py\
 ┣ 📜logging_policy.py :pencil2: \
 ┣ 📜loss.py :pencil2: \
 ┣ 📜model.py :pencil2: \
 ┣ 📜train.py :pencil2: \
 ┗ 📜utils.py

In the `ltr.dataset.FeatureExtraction.extract()` method, you are required to compute various features for a (query, document) pair. The list of features and their definitions can be found in the following table:

| Feature                | Definition |
|------------------------|------------|
| bm25| [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) score. Parameters: k1 = 1.5, b = 0.75 |
|query_term_coverage| Number of query terms in the document |
|query_term_coverage_ratio| Ratio of # query terms in the document to # query terms in the query  |
|stream_length| Length of document |
|idf| Sum of document frequencies. idf_smoothing = 0.5, formula: Log((N+1)/(df[term]+smoothing))| 
|sum_stream_length_normalized_tf| Sum over the ratios of each term to document length |
|min_stream_length_normalized_tf| Min over the ratios of each term to document length |
|max_stream_length_normalized_tf| Max over the ratios of each term to document length |
|mean_stream_length_normalized_tf| Mean over the ratios of each term to document length|
|var_stream_length_normalized_tf| Variance over the ratios of each term to document length |
|sum_tfidf| Sum of tf-idf|
|min_tfidf| Min of tf-idf|
|max_tfidf| Max of tf-idf |
|mean_tfidf| Mean of tf-idf |
|var_tfidf| Variance of tf-idf |

By completing all required implementation, you can achieve full marks on the implementation part, which can be computed by summing the points from passing each test found in [autograding.json](.github/classroom/autograding.json). The tests also depend on the results from each of your models saved in the [outputs](./outputs/) folder.

Make sure to __commit__ and __push__ the `.json` results files after training and running your models. For the analysis, you can base your observations on the reported metrics for the _test_ splits of the datasets that you train your models on.

**NOTICE THAT YOU NEED TO PUT YOUR IMPLEMENTATION BETWEEN `BEGIN \ END SOLUTION` TAGS!** All of the functions that need to be implemented in module directory files have a `#TODO` tag at the top of their definition. As always, it's fine to create additional code/functions/classes/etc. outside of these tags, but make sure not to change the behavior of the existing API.

### Analysis  

The goal of this analysis is to evaluate **Learning to Rank (LTR)** techniques by designing and implementing ranking models tailored to specific user needs in different retrieval scenarios. Your report should summarize key findings, justify improvements, and use visual aids such as tables and graphs.

You will first select one of the provided **retrieval scenarios**, each associated with a dataset and a specific information need. Based on this selection, you will propose and implement new **features and/or loss functions** to improve retrieval performance for that use case. Your implementation should be theoretically motivated, drawing from course materials or relevant research.

After implementing and evaluating your modifications, you will analyze their effects on ranking performance using **metrics such as NDCG, P@5, and Recall@20**. You should discuss whether your modifications led to expected improvements and provide theoretical reasoning to explain the outcomes. Additionally, a **per-query analysis** should be conducted to determine whether the new features or losses improved performance for specific types of queries or had a more general impact.

In the second part of this analysis, you will investigate the impact of **logging policy alterations and propensity clipping** on unbiased ranking models. You will modify the logging policy and clipping parameters, evaluate their effects, and discuss their influence on ranking fairness and performance. The results should be documented in the appropriate results table.

Support your findings with relevant tables and graphs. Submit the **compiled PDF** via **Canvas** before the deadline. Refer to the **complete instructions in the LaTeX template on Canvas** for detailed guidelines and submission requirements.


### General remarks on assignment organization
When you push to your repository on GitHub, unit tests will automatically run as in the previous parts. You can view the total points available and the points awarded to your current implementation following the same process before (i.e., click the 'x' or 'checkmark', and then view the output inside the 'Autograding' section).

Please note that you **SHOULD NOT** change params, seed(always=42), or any PATH variable given in the notebook for training LTR models with various loss functions, otherwise, you might lose points. We do NOT check the notebook; we ask this since we evaluate your results with the given parameters.

---
**Recommended Reading**:
  - Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. InProceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.
  - Christopher J Burges, Robert Ragno, and Quoc V Le. Learning to rank with nonsmooth cost functions. In Advances in neural information processing systems, pages 193–200, 2007
  - (Sections 1, 2 and 4) Christopher JC Burges. From ranknet to lambdarank to lambdamart: An overview. Learning, 11(23-581):81, 2010
  

Additional Resources: 
- This assignment requires knowledge of [PyTorch](https://pytorch.org/). If you are unfamiliar with PyTorch, you can go over [these series of tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) to get started.
- You can also refer to the [official documentation](https://pytorch.org/docs/stable/index.html) for more detailed explanations of the functions and classes used in this assignment.