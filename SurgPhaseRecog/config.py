class SurgConfig:
    def __init__(self, overrides=None):
        overrides = overrides or {}

        # Dataset
        self.dataset_config = {
            "llm":"e5-large",
            "model_name":"intfloat/multilingual-e5-large",
            "batch_size":32,
        }
        
        # Text Encoder
        self.embedder_config = {
            "llm":"e5-large",
            "model_name":"intfloat/multilingual-e5-large",
            "freeze":True,
            "freeze_layers":overrides.get("freeze_layers", 10),
        }
        
        # Classifier
        self.classifier_config = {
            "head": "single_layer",
            "input_dim": 1024, 
            "model_dim":overrides.get("model_dim", 1024),
            "n_classes": 8,
            "ff_dropout":overrides.get("ff_dropout", 0.3)
        }
        
        # Loss Function
        self.loss_config = {
            "function": "WCE",
            "focal_alpha":overrides.get("focal_alpha", 0.25),
            "focal_gamma":overrides.get("focal_gamma", 2),
            "ldam_m":overrides.get("ldam_m", 0.5),
            "ldam_s":overrides.get("ldam_2", 30)
        }
        
        # Training parameters
        self.params = {
            "lr": overrides.get("lr", 1e-4),
            "weight_decay":overrides.get("weight_decay", 1e-5),
            "patience_limit": 5,
            "epochs_limit": 500,
            "delta_escb": 0.001,
        }
        
        # Checks
        assert self.dataset_config["llm"] == self.embedder_config["llm"], "different llms are given"
        assert self.dataset_config["model_name"] == self.embedder_config["model_name"], "different models are given"