# Balancing the Scales: Using GPT-4 for Robust Data Augmentation

This repository accompanies the article **"Balancing the Scales: Using GPT-4 for Robust Data Augmentation"** by Mohammad Amin Samadi et al. The study explores using GPT-4 for data augmentation in collaborative problem-solving (CPS) to address class imbalance, specifically targeting the underrepresented "Cognitive Planning" (CP) class.

## Overview

This project uses three distinct GPT-4 prompting strategies for data augmentation:
- **Basic Prompting**: Paraphrasing with minimal prompt engineering.
- **Examples-Based Prompting**: Incorporates class-specific examples to guide the model.
- **Scenario-Based Prompting**: Generates original scenario-based utterances aligned with class definitions.

The generated samples are evaluated against a back-translation baseline (M2M100 model) on the following criteria:
1. **Content Consistency (CC)**: Retaining the original instance's meaning.
2. **Class Alignment (CA)**: Alignment with class-specific language patterns.
3. **Semantic Similarity (SS)**: Overlap in meaning between the resample and the original.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aminsmd/data-augmentation.git
   cd data-augmentation

2.	Install dependencies:
	```bash
	
	pip install -r requirements.txt


3.	Set up environment variables: Create a .env file in the project root and add your OpenAI API key:
	```bash

	OPENAI_API_KEY=your-api-key-here



Code Structure

	data-augmentation/
	├── data_generation/
	│   ├── run_augmentation.py    # Experiment runner
	│   ├── prompts.py             # Implementation of prompting strategies
	│   ├── data_augmentation.py   # Core augmentation logic
	│   └── backtranslation.py     # Back-translation baseline
	├── requirements.txt           # Project dependencies
	└── .env                       # Environment variables (API keys)

Usage Example

	from data_generation.run_augmentation import run_all_strategies
	import pandas as pd
	
	# Load your dataset
	data = pd.read_csv("your_data.csv")
	
	# Run all prompting strategies
	run_all_strategies(
	    data=data,
	    model_name="gpt-4-turbo-preview",
	    text_column="text",
	    class_column="label",
	    target_class="Cognitive Planning",
	    output_dir="results/augmentation"
	)

Evaluation Metrics
1. Content Consistency (CC): Measures semantic and structural fidelity.
2. Class Alignment (CA): Evaluates alignment with Cognitive Planning language patterns.
3. Semantic Similarity (SS): Quantifies thematic consistency with the source instance.

Results Summary

| Method          | CC (Mean, SD) | CA (Mean, SD) | SS (Mean, SD) |
|------------------|---------------|---------------|---------------|
| Back-translation | 0.427, 0.495  | 0.667, 0.472  | 0.761, 0.164  |
| Prompt 1         | 0.621, 0.486  | 0.721, 0.449  | 0.638, 0.172  |
| Prompt 2         | 0.688, 0.464  | 0.876, 0.330  | 0.671, 0.174  |
| Prompt 3         | N/A           | 0.997, 0.058  | 0.339, 0.080  |

Requirements
- Python 3.8+
- pandas >= 2.0.0
- langchain-openai >= 0.0.2
- python-dotenv >= 1.0.0
- openai >= 1.12.0

Citation

If you use this code in your research, please cite:

	Samadi, M. A., Shariff, D., Ibarra, R., & Nixon, N. (2025). Balancing the Scales: Using GPT-4 for Robust Data Augmentation. In Companion Proceedings of the 15th International Conference on Learning Analytics & Knowledge (LAK25).

We welcome contributions, suggestions, and collaborations to improve this repository! If you have any questions, ideas, or issues, feel free to:
- Open an issue in the repository.
- Start a discussion using the GitHub Discussions feature.
- Reach out via email (masamadi@uci.edu) for collaboration opportunities.

Let’s work together to advance robust data augmentation strategies in NLP and beyond!
