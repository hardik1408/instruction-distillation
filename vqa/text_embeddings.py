#!/usr/bin/env python3

import os
import json
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_subset(config):
    """Load the VQA subset data."""
    subset_path = config['output']['subset_path']
    with open(subset_path, 'r') as f:
        return json.load(f)

def precompute_question_embeddings(config):
    """
    Precompute question embeddings for all questions in the subset.
    """
    print("="*70)
    print("PRECOMPUTING QUESTION EMBEDDINGS")
    print("="*70)
    
    # Load subset
    subset_data = load_subset(config)
    questions_data = subset_data['data']
    
    print(f"\nTotal questions: {len(questions_data)}")
    
    # Load sentence transformer model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"\nLoading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Extract questions and question_ids
    question_ids = []
    questions = []
    
    for entry in questions_data:
        question_ids.append(str(entry['question_id']))
        questions.append(entry['question'])
    
    print(f"\nEncoding {len(questions)} questions...")
    
    # Compute embeddings in batches
    batch_size = 128
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Save embeddings
    output_path = config['output']['question_embeddings_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSaving embeddings to {output_path}...")
    np.save(output_path, {
        'embeddings': embeddings,
        'question_ids': question_ids,
        'questions': questions,
        'model': model_name,
        'embedding_dim': embeddings.shape[1]
    })
    
    print("\n" + "="*70)
    print("QUESTION EMBEDDINGS PRECOMPUTATION COMPLETE")
    print("="*70)
    print(f"Output file: {output_path}")
    print(f"Number of embeddings: {len(question_ids)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Verification
    print("\nVerifying saved embeddings...")
    loaded_data = np.load(output_path, allow_pickle=True).item()
    print(f"✓ Successfully loaded {loaded_data['embeddings'].shape[0]} embeddings")
    print(f"✓ Embedding dimension: {loaded_data['embedding_dim']}")
    print(f"✓ Model used: {loaded_data['model']}")
    
    # Show sample
    print("\nSample questions:")
    for i in range(min(5, len(questions))):
        print(f"  {i+1}. {questions[i]}")

def main():
    config = load_config()
    
    # Check if config has the output path for question embeddings
    if 'question_embeddings_path' not in config['output']:
        # Add default path
        config['output']['question_embeddings_path'] = 'data/question_embeddings.npy'
        print(f"Added default question_embeddings_path to config: {config['output']['question_embeddings_path']}")
    
    precompute_question_embeddings(config)

if __name__ == "__main__":
    main()