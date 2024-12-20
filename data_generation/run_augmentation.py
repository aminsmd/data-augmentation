import pandas as pd
from data_augmentation import augment_dataset, clean_generated_text
from prompts import PromptStrategy
from langchain_openai import ChatOpenAI
import os
from random import sample
from typing import Any
from dotenv import load_dotenv

def run_all_strategies(
    data: pd.DataFrame,
    model_name: str,
    text_column: str,
    class_column: str,
    target_class: str,
    output_dir: str,
    temperature: float = 0
) -> None:
    """Run data augmentation using all prompting strategies."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load environment variables and initialize model
    load_dotenv()
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # Run augmentation with each strategy
    for strategy in [PromptStrategy.BASIC, PromptStrategy.EXAMPLES, PromptStrategy.DETAILED]:
        print(f"Running augmentation with {strategy} strategy...")
        
        output_path = os.path.join(output_dir, f"augmented_{strategy}.csv")
        
        # Modify generate_paraphrases function to use current strategy
        def generate_with_strategy(text, llm, num_examples=3, example_pool=None):
            prompt = PromptStrategy.get_prompt(strategy, text, 
                                            random_examples='\n'.join(sample(example_pool, num_examples)) if example_pool else None)
            response = llm.predict(prompt)
            return clean_generated_text(response)
            
        # Override the generate_paraphrases function
        import data_augmentation
        data_augmentation.generate_paraphrases = generate_with_strategy
        
        # Run augmentation
        results = augment_dataset(
            data=data,
            llm=llm,
            text_column=text_column,
            class_column=class_column,
            target_class=target_class,
            output_path=output_path
        )
        
        print(f"Generated {len(results)} paraphrases")
        print(f"Results saved to {output_path}")
        print()

if __name__ == "__main__":
    # Load your data
    data = pd.read_csv("your_data.csv")
    
    # Run augmentation with all prompting strategies
    run_all_strategies(
        data=data,
        model_name="gpt-4-turbo-preview",  # or any other OpenAI model
        text_column="text",
        class_column="label",
        target_class="underrepresented_class",
        output_dir="results/augmentation"
    )