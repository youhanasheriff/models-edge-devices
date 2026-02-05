import fiftyone as fo
import fiftyone.zoo as foz

# Load the dataset we verified earlier (it should be cached)
# The verification script likely created 'quickstart-5'
try:
    if "quickstart-5" in fo.list_datasets():
        dataset = fo.load_dataset("quickstart-5")
    else:
        dataset = foz.load_zoo_dataset("quickstart", max_samples=5)
except Exception:
    # Fallback to loading fresh
    dataset = foz.load_zoo_dataset("quickstart", max_samples=5)

print(f"Loaded dataset: {dataset.name}")

# Launch the App
session = fo.launch_app(dataset)

print("\n" + "="*50)
print("FiftyOne App is running!")
print("Access it in your browser at: http://localhost:5151")
print("="*50 + "\n")

# Block execution so the app stays open
session.wait()
