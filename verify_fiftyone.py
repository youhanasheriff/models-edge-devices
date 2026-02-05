import fiftyone as fo
import fiftyone.zoo as foz

print("FiftyOne imported successfully.")
try:
    print("Loading quickstart dataset...")
    # Load just 5 samples to test, to save time/space
    dataset = foz.load_zoo_dataset("quickstart", max_samples=5)
    print("Dataset loaded.")
    # We won't launch the app here as it blocks, but just printing success is enough
    print("Verification complete: FiftyOne is ready.")
except Exception as e:
    print(f"Error: {e}")
