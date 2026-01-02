from pipeline import run_full_pipeline
from inference import run_inference
from skin_scores import compute_skin_scores, skin_detail_scale, skin_view_mode
from acne_detector import analyze_acne, count_pimples
from face_regions import region_scores
import cv2

def main():
    input_image = "sample_skin5.jpeg"
    processed_image = "final_skin.jpg"

    skin, mask, mode = run_full_pipeline(input_image)
    cv2.imwrite(processed_image, skin)
    print("Preprocessing done, saved as final_skin.jpg")
    
    # Check image distance
    sds = skin_detail_scale(skin, mask)
    view_mode = skin_view_mode(sds)
    print("\nSkin View Mode:", view_mode.upper())
    print("\nSkin Detail Scale:", sds, "/ 100")

    scores = compute_skin_scores(skin, mask)
    print("Skin Analysis Scores: ")
    for key, value in scores.items():
        if view_mode == "micro" and key in ["texture", "redness", "porosity"]:
            print(f"{key.capitalize()} : {value} / 100 (micro - detail sensitive, for close up image of skin)")
        else:
            print(f"{key.capitalize()} : {value} / 100")

    pimples = count_pimples(skin, mask)
    print("\nPimple Count:", pimples["pimple_count"])

    if mode == "face":
        region_s = region_scores(skin, mask)
        print("\nRegion-wise Facial Scores:")
        for region, scores in region_s.items():
            print(f"\n{region.upper()}")
            for key, value in scores.items():
                print(f" {key.capitalize()}: {value} / 100")
    else:
        print("\nRegion-wise Facial Scores: Skipped (Not a face image)")

    acne = analyze_acne(skin, mask)
    print("\nAcne Analysis:")
    print("Acne count: ", acne["acne_count"])
    print("Acne severity: ", acne["acne_severity"], "/ 100")

    result = run_inference(processed_image)

    print("Model Results:")
    print(result)

if __name__ == "__main__":
    main()