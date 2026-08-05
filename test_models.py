import pickle

models = [
    "models/cstick_model.pkl",
    "models/sisfall_model.pkl",
    "models/har70_model.pkl",
]

for path in models:
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        print(f"✅ {path} loaded successfully")
        print("Keys:", bundle.keys())
        print("Model name:", bundle.get("model_name", "Not specified"))
        print()

    except Exception as e:
        print(f"❌ {path} failed to load")
        print(e)
        print()