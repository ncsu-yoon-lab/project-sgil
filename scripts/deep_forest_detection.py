from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results
# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# sample_image_path = get_data("OSBS_029.png")
img = model.predict_image(path="../dataset/satellite_images/fayetteville_st.png")
plot_results(img)
