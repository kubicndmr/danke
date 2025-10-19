class SurgConfig:
    # Text Encoder BERT
    embedder_config = {
        "llm":"bert",
        "model_name":"bert-base-uncased",
        "freeze" : True,
        "freeze_layers": 5,
    }
    
    # Classifier
    classifier_config = {
        "head": "single_layer",
        "input_dim": 768, 
        "model_dim": 1024,
        "n_classes": 8,
        "ff_dropout": 0.3
    }