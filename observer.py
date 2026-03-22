from pathlib import Path

if __name__ == "__main__":
    
    logs_path = Path("logs")
    training = "K[None]Fold_nops[100-0]"

    for p in logs_path.iterdir():
        
        if p.name.startswith(training):
            log_file = p / "log.txt"

            with open(log_file) as f:
                lines = f.readlines()

            print()
            for r in lines[-5:]:
                if r != "":
                    print(r)


