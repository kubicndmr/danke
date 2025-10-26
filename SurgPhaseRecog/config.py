class SurgConfig:
    # Dataset
    dataset_config = {
        "llm":"bert",
        "model_name":"bert-base-uncased",
        "batch_size" : 512,
    }
    
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
    
    # Training parameters
    params = {
        "lr":1e-4,
        "weight_decay":1e-5,
        "batch_size":512,
        "patience_limit": 5,
        "epochs_limit": 500,
        "delta_escb": 0,
    }
    
    # Checks
    assert dataset_config["llm"] == embedder_config["llm"], "different llms are given"
    assert dataset_config["model_name"] == embedder_config["model_name"], "different models are given"