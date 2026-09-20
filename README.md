# Automated Support Ticket Classification and Tagging via NLP

## 1. Project Overview
This repository contains a Natural Language Processing (NLP) pipeline designed to automatically categorize customer support tickets. By implementing both zero-shot heuristic and few-shot machine learning techniques, the system effectively routes customer inquiries into specific domains, streamlining helpdesk workflows and reducing manual triage time.

## 2. Research Objectives
* **Automated Ticket Triage:** Develop a classification system to accurately assign support tickets to one of five core categories: Billing, Technical, Account, Shipping, or General.
* **Paradigm Comparison:** Evaluate the performance trade-offs between zero-shot (keyword-based) classification and few-shot (statistical machine learning) approaches on a constrained dataset.
* **Probabilistic Tagging:** Implement a Top-3 probability-ranked tag output to handle complex, multi-intent customer queries that span across multiple categories.

## 3. Dataset Characteristics
The model is trained and evaluated on a curated Support Ticket Dataset representing common helpdesk inquiries.
* **Volume:** 55 labeled tickets.
* **Distribution:** 5 discrete categories.
* **Integration:** Embedded directly within the notebook for immediate reproducibility without requiring external API calls or database connections.

## 4. Methodology & Architecture
The project explores two distinct NLP classification strategies:

* **Zero-Shot Classification (Heuristic):**
  * Utilizes domain-specific keyword and bigram matching.
  * Requires no prior training data or weight optimization. 
  * Highly effective for distinct-vocabulary categories (e.g., Shipping, Billing).
* **Few-Shot Classification (Statistical ML):**
  * **Feature Extraction:** Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to capture term importance and penalize common stop-word noise.
  * **Model:** Logistic Regression classifier trained on a small subset of labeled data.
  * **Inference:** Generates a softmax-style probability distribution across all classes to surface the Top-3 most relevant tags per ticket.

## 5. Results and Key Takeaways
* **Few-Shot Performance:** The TF-IDF + Logistic Regression pipeline achieved a peak **Accuracy of ~85%** and an **F1 Score of ~0.84**, significantly outperforming the zero-shot baseline on ambiguous queries (e.g., distinguishing between "Technical" and "Account" issues).
* **Zero-Shot Baseline:** Achieved **~76% Accuracy** and **~0.75 F1 Score**, proving viable as a lightweight, cold-start fallback mechanism.
* **Feature Importance:** Bigram features (e.g., "not working", "credit card") proved to be the strongest predictive indicators across all classes.

## 6. Production & Deployment Notes
* **Serialization:** The final inference pipeline is serialized via `joblib`, enabling rapid deployment into existing ticketing infrastructures (e.g., Zendesk, Jira, ServiceNow).
* **Future Scope:** 
  * Transition from sparse TF-IDF vectorization to dense contextual embeddings (e.g., Hugging Face, BERT) to capture deeper semantic relationships.
  * Expand the dataset size to accommodate deep sequence models.

## 7. Technology Stack
* **Language:** Python 3.10
* **Core Libraries:** scikit-learn, pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Serialization:** joblib
