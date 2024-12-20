import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re
from random import sample
from langchain_openai import ChatOpenAI

def clean_generated_text(text: str) -> List[str]:
    """Clean generated text by removing numbering and empty lines."""
    # Split text into lines
    lines = text.split('\n')
    
    # Remove numbering and whitespace
    pattern = re.compile(r'^\s*\d+\.\s*')
    cleaned = [pattern.sub('', line).strip() for line in lines]
    
    # Remove empty lines
    return [line for line in cleaned if line]

def generate_paraphrases(
    text: str,
    llm: ChatOpenAI,
    num_examples: int = 3,
    example_pool: Optional[List[str]] = None
) -> List[str]:
    """Generate paraphrases for given text using provided language model."""
    # Sample random examples if example pool is provided
    if example_pool:
        random_examples = '\n'.join(sample(example_pool, num_examples))
    else:
        random_examples = ''

    # Construct prompt
    prompt = f"""You have been assigned the task of data augmentation for an underrepresented class. 
    This involves generating new data belonging to that class by paraphrasing existing instances. 
    Here are three random examples that belong to the class.
    {random_examples}
    Please paraphrase the following text in 10 different ways, separating each paraphrase with a new line: {text}"""

    # Generate paraphrases
    response = llm.predict(prompt)
    
    # Clean and return results
    return clean_generated_text(response)

def augment_dataset(
    data: pd.DataFrame,
    llm: ChatOpenAI,
    text_column: str,
    class_column: str,
    target_class: str,
    num_examples: int = 3,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Augment dataset by generating paraphrases for underrepresented class.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing text data and class labels
    llm : ChatOpenAI
        Language model for text generation
    text_column : str
        Name of column containing text to paraphrase
    class_column : str
        Name of column containing class labels
    target_class : str
        Class label to augment
    num_examples : int
        Number of random examples to include in prompt
    output_path : Optional[str]
        Path to save results CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing original instances and generated paraphrases
    """
    # Get samples from target class
    samples = np.array(data[data[class_column] == target_class][text_column])
    example_pool = list(samples)
    
    # Initialize output dictionary
    out_dict = {}
    examples = []
    
    # Generate paraphrases for each sample
    for text in samples:
        paraphrases = generate_paraphrases(
            text,
            llm,
            num_examples=num_examples,
            example_pool=example_pool
        )
        out_dict[text] = paraphrases
        examples.append('\n'.join(sample(example_pool, num_examples)))
    
    # Create results dataframe
    results = {
        'Original_Instance': [],
        'Resample': [],
        'Score 1': [],
        'Score 2': []
    }
    
    for original, paraphrases in out_dict.items():
        for paraphrase in paraphrases:
            results['Original_Instance'].append(original)
            results['Resample'].append(paraphrase)
            results['Score 1'].append(None)
            results['Score 2'].append(None)
            
    df_results = pd.DataFrame(results)
    
    # Save results if path provided
    if output_path:
        df_results.to_csv(output_path, index=False)
        
        # Save examples used for each generation
        example_path = output_path.rsplit('.', 1)[0] + '_examples.npy'
        np.save(example_path, examples)
        
    return df_results

if __name__ == "__main__":
    # Example usage:
    """
    
    # Initialize your language model
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # Load your data
    data = pd.read_csv("your_data.csv")
    
    # Generate augmented dataset
    results = augment_dataset(
        data=data,
        llm=llm,
        text_column="text",
        class_column="label",
        target_class="underrepresented_class",
        output_path="results/augmented_data.csv"
    )
    """
    pass 