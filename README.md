# Towards Concept-Aware Large Language Models (EMNLP'23)
### [<b>Link to paper</b>](https://aclanthology.org/2023.findings-emnlp.877.pdf)

<ins>Abstract:</ins>
Concepts play a pivotal role in various human cognitive functions, including learning, reasoning and communication. However, there is very little work on endowing machines with the ability to form and reason with concepts. In particular, state-of-the-art large language models (LLMs) work at the level of tokens, not concepts. In this work, we analyze how well contemporary LLMs capture human concepts and their structure. We then discuss ways to develop concept-aware LLMs, taking place at different stages of the pipeline. We sketch a method for pretraining LLMs using concepts, and also explore the simpler approach that uses the output of existing LLMs. Despite its simplicity, our proof-of-concept is shown to better match human intuition, as well as improve the robustness of predictions. These preliminary results underscore the promise of concept-aware LLMs.


<ins>In this repository:</ins>

* <b>RQ1_2:</b> Implementation of the concept understanding of BERT, T5, GPT-3.5 and GPT-4.

* <b>RQ3:</b> Implementation of the concept-aware post-processing manipulation done using BERT -- Concept-BERT.




## Citation
Please cite our article as follows:
```bibtex
@inproceedings{shani-etal-2023-towards,
    title = "Towards Concept-Aware Large Language Models",
    author = "Shani, Chen  and
      Vreeken, Jilles  and
      Shahaf, Dafna",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.877",
    doi = "10.18653/v1/2023.findings-emnlp.877",
    pages = "13158--13170",
    abstract = "Concepts play a pivotal role in various human cognitive functions, including learning, reasoning and communication. However, there is very little work on endowing machines with the ability to form and reason with concepts. In particular, state-of-the-art large language models (LLMs) work at the level of tokens, not concepts. In this work, we analyze how well contemporary LLMs capture human concepts and their structure. We then discuss ways to develop concept-aware LLMs, taking place at different stages of the pipeline. We sketch a method for pretraining LLMs using concepts, and also explore the simpler approach that uses the output of existing LLMs. Despite its simplicity, our proof-of-concept is shown to better match human intuition, as well as improve the robustness of predictions. These preliminary results underscore the promise of concept-aware LLMs.",
}
