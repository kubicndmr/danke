class SurgConfig:
    # Text Encoder BERT
    embedder_config = {
        "llm":"bert",
        "model_name":"bert-base-uncased",
        "freeze" : True,
        "freeze_layers": 10,
    }
    
    # Classifier
    classifier_config = {
        "head": "single_layer",
        "input_dim": 768, 
        "model_dim": 1024,
        "n_classes": 8,
        "ff_dropout": 0.3
    }
    
    # Loss Function
    loss_config = {
        "function": "WCE",
        "focal_alpha":0.25,
        "focal_gamma":2,
        "ldam_m":0.5,
        "ldam_s":30
    }