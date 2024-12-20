import pandas as pd
from typing import List, Dict, Optional
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import os

class BackTranslator:
    # Dictionary of supported languages and their codes
    LANGUAGES = {
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Italian": "it",
        "Dutch": "nl",
        "Portuguese": "pt",
        "Russian": "ru",
        "Chinese": "zh",
        "Arabic": "ar",
        "Japanese": "ja"
    }
    
    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        """Initialize the backtranslation model and tokenizer."""
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        
    def translate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate texts from source language to target language."""
        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        generated_tokens = self.model.generate(
            **encoded, 
            forced_bos_token_id=self.tokenizer.get_lang_id(target_lang)
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    def backtranslate(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Perform backtranslation through an intermediate language."""
        # Translate to target language
        intermediate = self.translate(texts, source_lang, target_lang)
        
        # Translate back to source language
        return self.translate(intermediate, target_lang, source_lang)

def augment_dataset(
    data: pd.DataFrame,
    text_column: str,
    source_lang: str = "en",
    languages: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    model_name: str = "facebook/m2m100_418M"
) -> pd.DataFrame:
    """
    Augment dataset using backtranslation through multiple languages.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing text data
    text_column : str
        Name of column containing text to augment
    source_lang : str
        Source language code (default: "en")
    languages : Optional[Dict[str, str]]
        Dictionary of language names and codes to use (default: all supported languages)
    output_path : Optional[str]
        Path to save results CSV
    model_name : str
        Name of the M2M100 model to use
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing original texts and their backtranslations
    """
    # Initialize backtranslator
    translator = BackTranslator(model_name)
    
    # Use default languages if none provided
    if languages is None:
        languages = BackTranslator.LANGUAGES
    
    # Get texts to augment
    texts = data[text_column].tolist()
    
    # Initialize results list
    augmented_data = []
    
    # Apply backtranslation for each language
    for language, code in languages.items():
        print(f"Applying backtranslation with {language}...")
        augmented_texts = translator.backtranslate(texts, source_lang, code)
        
        for original, augmented in zip(texts, augmented_texts):
            augmented_data.append({
                "Original_Text": original,
                "Language": language,
                "Augmented_Text": augmented
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(augmented_data)
    
    # Save results if path provided
    if output_path:
        df_results.to_csv(output_path, index=False)
    
    return df_results

if __name__ == "__main__":
    # Example usage:
    data = pd.read_csv("your_data.csv")
    
    # Optionally specify subset of languages
    target_languages = {
        "French": "fr",
        "German": "de",
        "Spanish": "es"
    }
    
    results = augment_dataset(
        data=data,
        text_column="text",
        languages=target_languages,
        output_path="results/backtranslated_texts.csv"
    ) 