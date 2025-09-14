import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch.nn as nn
from torch.nn import functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalPhaseDataset(Dataset):
    """Custom dataset for surgical phase classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx]) - 1  # Convert to 0-based indexing
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SurgicalPhaseClassifier(nn.Module):
    """Classification model based on multilingual-e5-large."""
    
    def __init__(self, model_name, num_classes=8, dropout_rate=0.3):
        super(SurgicalPhaseClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Load the pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get embeddings from the encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

def load_pickle_files(data_directory):
    """Load all pickle files from the specified directory."""
    data_dir = Path(data_directory)
    all_data = []
    
    for pkl_file in data_dir.glob("*.pkl"):
        logger.info(f"Loading {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    all_data.append(data)
                else:
                    logger.warning(f"File {pkl_file} does not contain a DataFrame")
        except Exception as e:
            logger.error(f"Error loading {pkl_file}: {e}")
    
    if not all_data:
        raise ValueError("No valid pickle files found in the directory")
    
    # Concatenate all dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined_data)} total samples from {len(all_data)} files")
    
    return combined_data

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    # Configuration
    MODEL_NAME = "intfloat/multilingual-e5-large"
    DATA_DIRECTORY = "/DATA/kubi/Dataset/SynPoCaP"  # Change this to your data directory
    OUTPUT_DIR = "./surgical_phase_model"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading pickle files...")
    df = load_pickle_files(DATA_DIRECTORY)
    
    # Validate required columns
    required_columns = ['Text', 'Phase_Label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Remove any rows with missing values
    df = df.dropna(subset=required_columns)
    logger.info(f"Dataset shape after cleaning: {df.shape}")
    
    # Validate phase labels
    unique_labels = sorted(df['Phase_Label'].unique())
    logger.info(f"Unique phase labels: {unique_labels}")
    
    if not all(isinstance(label, (int, np.integer)) and 1 <= label <= 8 for label in unique_labels):
        logger.warning("Phase labels should be integers from 1 to 8")
    
    # Print class distribution
    label_counts = df['Phase_Label'].value_counts().sort_index()
    logger.info("Class distribution:")
    for label, count in label_counts.items():
        logger.info(f"  Phase {label}: {count} samples")
    
    # Split the data
    texts = df['Text'].tolist()
    labels = df['Phase_Label'].tolist()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = SurgicalPhaseDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = SurgicalPhaseDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = SurgicalPhaseDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    # Initialize model
    logger.info("Loading model...")
    model = SurgicalPhaseClassifier(MODEL_NAME, num_classes=8)
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    # Generate detailed classification report
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
    
    # Convert back to 1-based indexing for report
    all_predictions = [p + 1 for p in all_predictions]
    all_labels = [l + 1 for l in all_labels]
    
    # Print detailed classification report
    target_names = [f'Phase_{i}' for i in range(1, 9)]
    report = classification_report(all_labels, all_predictions, target_names=target_names)
    logger.info("Detailed Classification Report:")
    logger.info(f"\n{report}")
    
    # Save classification report
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)
    
    logger.info(f"Training completed! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()