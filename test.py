
Since Root Cause Level 2 (Lv2) consists of 13 labels, making it a challenging multi-class classification task, we implemented a hierarchical classification approach to enhance prediction performance. This method leverages prior knowledge from Root Cause Level 1 (Lv1), where each Lv2 label belongs to a specific Lv1 category, effectively functioning as a sub-label.

Our approach follows a two-step process. First, we predict the Root Cause Lv1 labels. Then, we use the predicted Lv1 label as a constraint, guiding the LLM to select the most appropriate Lv2 label only within the corresponding Lv1 category. By narrowing down the prediction scope and reducing the complexity of the classification task, this hierarchical method significantly improves accuracy.

Additionally, this approach aligns with the natural structure of risk events, ensuring that the model learns meaningful relationships between different classification levels. It also enhances interpretability by enforcing logical consistency in predictions, making it a practical solution for real-world risk event classification.
