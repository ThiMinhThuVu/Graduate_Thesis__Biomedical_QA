The question answering model is developed based on multi-hop search using search engines, combined with the reasoning capabilities of LLMs:
(1) Processing question: Simplify question and identify question type
(2) Classify question into 3 mode: direct, sequential, parallel using Bert-based classification
(3) Answer Generation:
    - Retrieve initial evidence from Open Web and Wikipedia.
    - Cluster retrieved contexts and use cluster centers to collect detailed Wikipedia evidence.
    - For each cluster, invoke LLM to generate candidate answers with traceable explanations.
    - Perform alias mapping in vector space to group equivalent entities and select the final answer.
(4) Answer Normalization: Apply alias resolution step to produce the final answer.