
Earlier this year, our teams jointly optimized the infrastructure used to train deep learning sequence models. Specifically, we aligned on standardizing data loading pipelines, introducing streaming data ingestion using tools like litdata, and configuring consistent Distributed Data Parallel (DDP) training setups across teams.

As a result of this collaboration:

Training run time improved by ~30-40% on large-scale sequence models due to faster data loading, reduced I/O bottlenecks, and better GPU utilization.

We implemented a shared, scalable training framework that supported multi-node and multi-GPU training, enabling teams to train models at previously infeasible scales.

Our work reduced duplicated engineering efforts and allowed others to onboard faster — multiple teams adopted this shared framework in their production or research training jobs.

This collaboration demonstrates strong horizontal impact: rather than improving just our own team’s modeling workflows, we created reusable tooling and infrastructure that benefited adjacent teams working on NLP, fraud detection, and time-series forecasting. It also helped establish best practices for sequence modeling within the org and reinforced the culture of open engineering knowledge-sharing.
