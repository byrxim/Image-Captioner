from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import easyocr
import cv2
import textwrap

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image_path):
    images = []
    for image_path in  image_path:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
    
    pixel_values = feature_extractor(
        images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    preds = [pred.strip() for pred in preds]
    return preds

f=str((predict_caption(['images.jpg'])))
f="Caption:"+f


imS = cv2.imread("images.jpg")
reader = easyocr.Reader(['en'])
Result = reader.readtext(imS)

for (bbox, text, prob) in Result: 
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    cv2.rectangle(imS, tl, br, (255, 255, 0), 2)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

    imS = cv2.rectangle(imS, (tl[0], tl[1] - 20), (tl[0] + w, tl[1]), (200, 200, 255), -1)
    cv2.putText(imS, text, (tl[0], tl[1] - 5),
      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
size, _=cv2.getTextSize(f, cv2.FONT_HERSHEY_SIMPLEX,0.5 ,1)
width, height = size

imS = cv2.rectangle(imS, (0, 0), (width, height+2), (200, 200, 255), -1)
cv2.putText(imS, f, (0, height), cv2.FONT_HERSHEY_SIMPLEX,0.5 , (0,0,0), 1, lineType = cv2.LINE_AA)
print(f)
cv2.imshow("Result", imS)
cv2.waitKey(0)
cv2.destroyAllWindows()
