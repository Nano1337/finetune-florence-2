import concurrent.futures
import io

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random

# TODO: read the Flan paper to figure out how to diversify question templates
question_templates = [
    'Is the caption "{}" accurate and detailed in describing the image?',
    'Does the description "{}" match what you see in the image?',
    'Is there a strong correspondence between the image and the text "{}"?',
    'Does the image contain all the key elements mentioned in "{}"?',
    'Would you say that "{}" provides a faithful representation of the image?',
    'Is the statement "{}" consistent with the visual content of the image?',
    'Can all the details mentioned in "{}" be verified in the image?',
    'Does "{}" accurately capture the main subject or focus of the image?',
     'Is "{}" comprehensive in covering all major aspects of the image?',
     'Is "{}" entirely consistent with what you see in the image?', 
]

yesono_template = [
    'Please answer using a single word: yes or no.',
    'Please answer yes or no.',
    'Respond with either "yes" or "no".',
    'Give a one-word answer: yes or no.',
    'Answer with just "yes" or "no".',
    'Provide a binary response: yes/no.',
    'Reply with a simple yes or no.',
    'Use only "yes" or "no" in your response.',
    'Limit your answer to yes or no.',
    'Respond briefly with yes or no.',
    'Give a concise yes or no answer.',
    'Answer affirmatively or negatively: yes/no.',
]


class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text
    
class FilteringVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        data_path = "/home/haoli/VLM-prune/data/combined_florence_data.parquet"
        full_data = pd.read_parquet(data_path)
        
        # Shuffle the data
        full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(full_data) * 0.99)
        
        # Split the data
        if split == 'train':
            self.data = full_data.iloc[:split_idx]
        elif split == 'val':
            self.data = full_data.iloc[split_idx:]
        else:
            raise ValueError("Split must be either 'train' or 'val'")
        
        self.task_prompt = "<FILTER>"
    
    def sample_template(self, caption):

        # random order and question type for robustness
        random_boolean = random.choice([True, False])
        question_template = random.choice(question_templates)
        question_template = question_template.format(caption)
        yesono_template = random.choice(yesono_template)

        if random_boolean:
            return self.task_prompt + " " + question_template + " " + yesono_template
        else:
            return self.task_prompt + " " + yesono_template + " " + question_template

    def __getitem__(self, idx): 
        row = self.data.iloc[idx]
        question = self.sample_template(row['caption'])
        answer = row['label']
        image_path = row['file_path']
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image


class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        self.task_prompt = "<DocVQA>"

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = self.correct_casing_finqa(first_answer)
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image
    

class TheCauldronDataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.images_df, self.texts_df = self.load_all_configs(split)
        self.task_prompt = "<VQA>"

    def __len__(self):
        return len(self.texts_df)
    
    def load_config(self, config_name, split):
        print(f"Loading config: {config_name}")
        dataset = load_dataset("HuggingFaceM4/the_cauldron", config_name, split=split)
        print(f"Finished loading config: {config_name}")

        df_data = dataset.to_pandas()

        # Create the images DataFrame
        df_images = df_data[['images']].copy()
        df_images['image_index'] = df_images.index

        # Explode the texts into separate rows and create a DataFrame
        df_texts = df_data[['texts']].explode('texts').reset_index()
        df_texts.rename(columns={'index': 'image_index'}, inplace=True)

        # Extract 'user', 'assistant', and 'source' from the 'texts' column
        df_texts['question'] = df_texts['texts'].apply(lambda x: x.get('user'))
        df_texts['answer'] = df_texts['texts'].apply(lambda x: x.get('assistant'))
        df_texts['source'] = df_texts['texts'].apply(lambda x: x.get('source'))

        # Drop the original 'texts' column
        df_texts.drop(columns=['texts'], inplace=True)

        # Copy the 'source' column to the images df, using the first source per image index
        df_images = df_images.merge(df_texts[['image_index', 'source']], on='image_index', how='left')
        print(f"Finished processing config: {config_name}")

        return df_images, df_texts

    def load_all_configs(self, split):
        cauldron_config_names = get_dataset_config_names("HuggingFaceM4/the_cauldron")

        images_dfs = []
        texts_dfs = []

        # Use ThreadPoolExecutor for parallel processing and tqdm for progress tracking
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  # Limit the number of workers
            with tqdm(total=len(cauldron_config_names), desc="Total Progress") as total_pbar:
                futures = {executor.submit(self.load_config, config_name, split): config_name for config_name in cauldron_config_names}
                for future in concurrent.futures.as_completed(futures):
                    config_name = futures[future]
                    try:
                        df_images, df_texts = future.result()
                        images_dfs.append(df_images)
                        texts_dfs.append(df_texts)
                    except Exception as exc:
                        print(f"{config_name} generated an exception: {exc}")
                    total_pbar.update(1)

        # Merge all the loaded DataFrames
        print("Merging DataFrames...")
        merged_images_df = pd.concat(images_dfs, ignore_index=True)
        merged_texts_df = pd.concat(texts_dfs, ignore_index=True)
        print("Finished merging DataFrames")

        return merged_images_df, merged_texts_df

    def __getitem__(self, idx):
        example = self.texts_df.iloc[idx]
        question = example["question"]
        answer = example["answer"]
        source = example["source"]
        image_idx = example["image_index"]

        image_data = self.images_df.loc[(self.images_df['image_index'] == image_idx) & (self.images_df['source'] == source), 'images'].values[0][0]['bytes'] 
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image
