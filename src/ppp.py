import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import classification_report as ner_classification_report
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and parameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = "bert-base-uncased"

# Load and prepare datasets (example format)
# You'll need to replace this with your actual dataset loading code
"""
Sample format for your dataset:
{
    "text": "A massive earthquake hit Mexico City yesterday, causing widespread damage.",
    "is_disaster": 1,  # 1 for disaster, 0 for non-disaster
    "disaster_type": "earthquake",  # flood, earthquake, hurricane, etc.
    "locations": [(20, 30, "Mexico City")]  # (start_idx, end_idx, location_name)
}
"""

# Step 1: Create dataset classes for each task

# Dataset for disaster detection (binary classification)
class DisasterDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Dataset for disaster type classification (multi-class)
class DisasterTypeDataset(Dataset):
    def __init__(self, texts, labels, label_map, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_map[self.labels[idx]]  # Convert label to index
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Dataset for NER (location extraction)
class LocationNERDataset(Dataset):
    def __init__(self, texts, location_tags, tokenizer, max_len):
        self.texts = texts
        self.location_tags = location_tags  # List of token-level tags in IOB format
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {'O': 0, 'B-LOC': 1, 'I-LOC': 2}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        location_tags = self.location_tags[idx]
        
        # Tokenize with word-to-token alignment
        tokenized = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].flatten()
        attention_mask = tokenized['attention_mask'].flatten()
        offset_mapping = tokenized['offset_mapping'].squeeze()
        
        # Convert character-level tags to token-level tags (simplified version)
        # In a real implementation, you'll need more sophisticated alignment code
        labels = torch.ones(input_ids.shape, dtype=torch.long) * -100  # -100 is ignored in loss
        
        # This is a simplified tag assignment - you'll need proper alignment in a real case
        token_labels = [self.label_map.get(tag, 0) for tag in location_tags]
        labels[:len(token_labels)] = torch.tensor(token_labels)
        
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Step 2: Define models for each task

# Model for disaster detection (binary classification)
class DisasterDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DisasterDetectionModel, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Model for disaster type classification (multi-class)
class DisasterTypeModel(nn.Module):
    def __init__(self, num_classes):
        super(DisasterTypeModel, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Model for location NER
class LocationNERModel(nn.Module):
    def __init__(self, num_labels=3):  # O, B-LOC, I-LOC
        super(LocationNERModel, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

# Step 3: Training functions

# Generic training function for classification models
def train_classification_model(model, train_dataloader, val_dataloader, epochs, device):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * epochs
    )
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                _, preds = torch.max(outputs, dim=1)
                
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
        
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Validation Accuracy: {val_acc}")
        print(f"Validation F1: {val_f1}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save model checkpoint
            # torch.save(model.state_dict(), f"best_model_{epoch}.pt")
            
    return model

# Training function for NER model
def train_ner_model(model, train_dataloader, val_dataloader, epochs, device):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * epochs
    )
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Reshape for loss calculation
            active_loss = attention_mask.view(-1) == 1
            active_logits = outputs.view(-1, 3)  # 3 = num_labels
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(-100).type_as(labels)
            )
            
            loss = F.cross_entropy(active_logits, active_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        
        # Evaluation - simplified version
        model.eval()
        with torch.no_grad():
            # You'd typically compute token-level metrics here
            # For simplicity, we're just showing the structure
            pass
            
    return model

# Step 4: Pipeline for prediction

class DisasterPipeline:
    def __init__(self, 
                 detection_model,
                 type_model,
                 ner_model,
                 tokenizer,
                 disaster_type_map,
                 max_len=128,
                 device='cpu'):
        
        self.detection_model = detection_model.to(device)
        self.type_model = type_model.to(device)
        self.ner_model = ner_model.to(device)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.disaster_type_map = disaster_type_map
        self.id_to_type = {v: k for k, v in disaster_type_map.items()}
        self.id_to_ner = {0: 'O', 1: 'B-LOC', 2: 'I-LOC'}
        
    def predict(self, text):
        # Step 1: Detect if it's a disaster
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        # Detection prediction
        self.detection_model.eval()
        with torch.no_grad():
            detection_outputs = self.detection_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            _, detection_pred = torch.max(detection_outputs, dim=1)
            is_disaster = detection_pred.item() == 1
        
        if not is_disaster:
            return {
                'is_disaster': False,
                'disaster_type': None,
                'locations': []
            }
        
        # Step 2: Classify the disaster type
        self.type_model.eval()
        with torch.no_grad():
            type_outputs = self.type_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            _, type_pred = torch.max(type_outputs, dim=1)
            disaster_type = self.id_to_type[type_pred.item()]
        
        # Step 3: Extract locations
        self.ner_model.eval()
        tokenized = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        offset_mapping = tokenized['offset_mapping'].squeeze().tolist()
        
        with torch.no_grad():
            ner_outputs = self.ner_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            ner_preds = torch.argmax(ner_outputs, dim=2).squeeze().tolist()
        
        # Convert predictions to locations
        locations = []
        current_loc = []
        current_start = None
        
        for i, (offset, pred, mask) in enumerate(zip(offset_mapping, ner_preds, attention_mask[0].tolist())):
            # Skip special tokens and padding
            if mask == 0 or offset == [0, 0]:
                continue
                
            tag = self.id_to_ner[pred]
            
            if tag == 'B-LOC':
                # End previous location if any
                if current_loc:
                    loc_text = ' '.join(current_loc)
                    locations.append({
                        'text': loc_text,
                        'start': current_start,
                        'end': offset_mapping[i-1][1]
                    })
                    current_loc = []
                
                # Start new location
                current_start = offset[0]
                current_loc.append(text[offset[0]:offset[1]])
                
            elif tag == 'I-LOC' and current_loc:
                # Continue current location
                current_loc.append(text[offset[0]:offset[1]])
                
            elif current_loc:
                # End location
                loc_text = ' '.join(current_loc)
                locations.append({
                    'text': loc_text,
                    'start': current_start,
                    'end': offset_mapping[i-1][1]
                })
                current_loc = []
                current_start = None
        
        # Add final location if any
        if current_loc:
            loc_text = ' '.join(current_loc)
            locations.append({
                'text': loc_text,
                'start': current_start,
                'end': offset_mapping[len(offset_mapping)-1][1]
            })
        
        return {
            'is_disaster': True,
            'disaster_type': disaster_type,
            'locations': locations
        }

# Step 5: Example usage (creating synthetic data for demonstration)

def create_sample_data():
    # Create synthetic data for demonstration
    texts = [
        "A massive earthquake hit Mexico City yesterday, causing widespread damage.",
        "Severe flooding in Bangkok has displaced thousands of residents.",
        "Hurricane Maria devastated Puerto Rico with winds of over 155 mph.",
        "I just watched a great movie about natural disasters.",
        "The basketball game was canceled due to bad weather."
    ]
    
    is_disaster = [1, 1, 1, 0, 0]
    
    disaster_types = ["earthquake", "flood", "hurricane", None, None]
    
    # Simplified location tags (in real implementation, you'd use character-level tags)
    location_tags = [
        ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ]
    
    return texts, is_disaster, disaster_types, location_tags

def main():
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create sample data
    texts, is_disaster, disaster_types, location_tags = create_sample_data()
    
    # Create disaster type mapping
    disaster_type_list = list(set([dt for dt in disaster_types if dt is not None]))
    disaster_type_map = {disaster_type: i for i, disaster_type in enumerate(disaster_type_list)}
    
    # Create datasets for each task
    # 1. Disaster Detection
    train_texts_detect, val_texts_detect, train_labels_detect, val_labels_detect = train_test_split(
        texts, is_disaster, test_size=0.2, random_state=42
    )
    
    train_dataset_detect = DisasterDetectionDataset(
        train_texts_detect, train_labels_detect, tokenizer, MAX_LEN
    )
    
    val_dataset_detect = DisasterDetectionDataset(
        val_texts_detect, val_labels_detect, tokenizer, MAX_LEN
    )
    
    # Create data loaders
    train_dataloader_detect = DataLoader(
        train_dataset_detect, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    
    val_dataloader_detect = DataLoader(
        val_dataset_detect, batch_size=VALID_BATCH_SIZE, shuffle=False
    )
    
    # Initialize and train disaster detection model
    detection_model = DisasterDetectionModel().to(device)
    
    print("Training disaster detection model...")
    detection_model = train_classification_model(
        detection_model, train_dataloader_detect, val_dataloader_detect, EPOCHS, device
    )
    
    # 2. Disaster Type Classification (only for disaster samples)
    disaster_texts = [text for text, is_d in zip(texts, is_disaster) if is_d == 1]
    disaster_types_subset = [dt for dt, is_d in zip(disaster_types, is_disaster) if is_d == 1]
    
    train_texts_type, val_texts_type, train_labels_type, val_labels_type = train_test_split(
        disaster_texts, disaster_types_subset, test_size=0.2, random_state=42
    )
    
    train_dataset_type = DisasterTypeDataset(
        train_texts_type, train_labels_type, disaster_type_map, tokenizer, MAX_LEN
    )
    
    val_dataset_type = DisasterTypeDataset(
        val_texts_type, val_labels_type, disaster_type_map, tokenizer, MAX_LEN
    )
    
    train_dataloader_type = DataLoader(
        train_dataset_type, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    
    val_dataloader_type = DataLoader(
        val_dataset_type, batch_size=VALID_BATCH_SIZE, shuffle=False
    )
    
    # Initialize and train disaster type model
    type_model = DisasterTypeModel(len(disaster_type_map)).to(device)
    
    print("Training disaster type classification model...")
    type_model = train_classification_model(
        type_model, train_dataloader_type, val_dataloader_type, EPOCHS, device
    )
    
    # 3. Location NER
    # For simplicity, we'll use the same split as before
    train_texts_ner, val_texts_ner, train_tags_ner, val_tags_ner = train_test_split(
        texts, location_tags, test_size=0.2, random_state=42
    )
    
    train_dataset_ner = LocationNERDataset(
        train_texts_ner, train_tags_ner, tokenizer, MAX_LEN
    )
    
    val_dataset_ner = LocationNERDataset(
        val_texts_ner, val_tags_ner, tokenizer, MAX_LEN
    )
    
    train_dataloader_ner = DataLoader(
        train_dataset_ner, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    
    val_dataloader_ner = DataLoader(
        val_dataset_ner, batch_size=VALID_BATCH_SIZE, shuffle=False
    )
    
    # Initialize and train NER model
    ner_model = LocationNERModel().to(device)
    
    print("Training location extraction model...")
    ner_model = train_ner_model(
        ner_model, train_dataloader_ner, val_dataloader_ner, EPOCHS, device
    )
    
    # Create the pipeline
    pipeline = DisasterPipeline(
        detection_model=detection_model,
        type_model=type_model,
        ner_model=ner_model,
        tokenizer=tokenizer,
        disaster_type_map=disaster_type_map,
        device=device
    )
    
    # Example usage
    text = "A severe hurricane has devastated parts of Florida with winds exceeding 130 mph."
    result = pipeline.predict(text)
    print("Prediction result:")
    print(f"Is disaster: {result['is_disaster']}")
    print(f"Disaster type: {result['disaster_type']}")
    print(f"Locations: {result['locations']}")

if __name__ == "__main__":
    main()

# To train this in your environment, you would need:
# 1. A properly formatted dataset with disaster binary labels, type labels, and NER annotations
# 2. More sophisticated NER alignment code for production use
# 3. Additional evaluation metrics and model checkpointing
# 4. Proper hyperparameter tuning