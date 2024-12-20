from typing import Optional, List

def get_basic_prompt(text: str) -> str:
    """Basic prompt that just requests paraphrasing."""
    return f"paraphrase the following text in 10 different ways, separated by \n (Only include the sentences, no numbers needed): {text}"

def get_examples_prompt(text: str, random_examples: Optional[str] = None) -> str:
    """Prompt that includes random examples from the same class."""
    return f"""You have been assigned the task of data augmentation for an underrepresented class. This involves generating new data belonging to that class by paraphrasing existing instances. Here are three random examples that belong to the class.
    {random_examples}
    Please paraphrase the following text in 10 different ways, separating each paraphrase with a new line: {text}"""

def get_detailed_prompt(text: str, random_examples: Optional[str] = None) -> str:
    """Detailed prompt with examples and specific instructions."""
    return f"""You are tasked with generating variations of text that belong to the same class as the input. 
    This is for data augmentation of an underrepresented class.
    
    Here are some examples from this class to help you understand the style and content:
    {random_examples}
    
    Please generate 10 different paraphrases of the following text. Each paraphrase should:
    - Maintain the same core meaning and intent
    - Use natural, conversational language
    - Vary in sentence structure and word choice
    - Be separated by new lines
    
    Text to paraphrase: {text}"""

class PromptStrategy:
    BASIC = "basic"
    EXAMPLES = "examples" 
    DETAILED = "detailed"
    
    @staticmethod
    def get_prompt(strategy: str, text: str, random_examples: Optional[str] = None) -> str:
        if strategy == PromptStrategy.BASIC:
            return get_basic_prompt(text)
        elif strategy == PromptStrategy.EXAMPLES:
            return get_examples_prompt(text, random_examples)
        elif strategy == PromptStrategy.DETAILED:
            return get_detailed_prompt(text, random_examples)
        else:
            raise ValueError(f"Unknown prompt strategy: {strategy}") 