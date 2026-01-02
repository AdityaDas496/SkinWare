from preprocessing.preprocess import preprocess_image
from models.cnn_model import SkinCNN
from models.clip_model import CLIPDescriber
from models.fusion import fuse_results

def run_inference(image_path):
    prompts = []
    file = open("prompts/skin_descriptors.txt", "r")
    for line in file:
        prompts.append(line.strip())

    file.close()

    image_tensor = preprocess_image(image_path)

    cnn = SkinCNN()
    dryness, texture, redness, pigmentation = cnn.infer(image_tensor)
    clip_model = CLIPDescriber()
    clip_tags = clip_model.describe(image_path, prompts)

    result = fuse_results(
        dryness,
        texture,
        redness,
        pigmentation,
        clip_tags
    )
    return result